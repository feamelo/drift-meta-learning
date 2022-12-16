import sys

sys.path.insert(0,'..')
sys.path.insert(0,'../..')

import pandas as pd
import numpy as np
import json
from meta_learning import MetaModel


# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')


PERF_METRICS = [
    "precision",
    "recall",
    "f1-score",
    "auc",
    "kappa",
]


class DriftContributionGenerator():
    def __init__(
        self,
        base_model: str,
        dataset_name: str,
        train_batch_size: str = 200,
        n_models: int = 1,
        output_filename: str=None,
        meta_model_params: dict={},
    ):
        self.train_batch_size = train_batch_size
        self.base_model = base_model
        self.dataset_name = dataset_name
        self.n_models = n_models
        self.meta_model_params = meta_model_params

        if not output_filename:
            output_filename = f"base_model: {self.base_model} - dataset: {self.dataset_name}"
        self.output_filename = output_filename

    def _load_metabase(self) -> None:
        filename = f"basemodel: {self.base_model} - metric: kappa - dataset: {self.dataset_name}"
        self.metabase = pd.read_csv(f"metabases/{filename} - with_drift_metrics.csv")

    def _create_results_df(self) -> None:
        self.metrics = list(set(self.metabase.columns).intersection(set(PERF_METRICS)))
        final_cols = self.metrics + [f"last_{metric}" for metric in self.metrics]
        self.results = self.metabase[final_cols]
        for metric in self.metrics:
            for i in range(self.n_models):
                self.results[f"{metric}_pred_{i}_with_drift"] = 0
                self.results[f"{metric}_pred_{i}_without_drift"] = 0

    def _create_meta_models(self):
        self.meta_models = {}
        for metric in self.metrics:
            self.meta_models[metric] = {
                "with_drift": [MetaModel(**self.meta_model_params) for _ in range(self.n_models)],
                "without_drift": [MetaModel() for _ in range(self.n_models)],
            }

    def _imp_dict(self, meta_model):
        model = meta_model.model
        importances = np.array(model.feature_importances_, dtype=float)
        return dict(zip(model.feature_name_, importances))

    def _get_importances(self):
        importances = {}
        for metric, values in self.meta_models.items():
            importances[metric] = {}
            for drift_flag, models in values.items():
                importances[metric][drift_flag] = [self._imp_dict(m) for m in models]
        return importances

    def _get_drift_cols(self) -> list:
        drift_suffixes = [
            "psi_",
            "overlap_",
            "omv_pth",
            "dc_accuracy",
            "dc_drift_flag",
            "_ks_statistic",
            "_ks_pvalue",
            "sqsi_drift_flag",
            "distance_class_",
            "u_detect_drift_flag",
            "predict",
            "last",
        ]

        self.drift_cols = []
        for col in self.metabase.columns:
            if any(ds in col for ds in drift_suffixes):
                self.drift_cols.append(col)

    def _train_metamodels(self, batch: pd.DataFrame):
        for metric in self.metrics:
            features = batch.drop(self.metrics, axis=1)
            non_drift_features = features.drop(self.drift_cols, axis=1)
            target = batch[metric]
            for i in range(self.n_models):
                self.meta_models[metric]["with_drift"][i].fit(features, target)
                self.meta_models[metric]["without_drift"][i].fit(non_drift_features, target)

    def _make_prediction(self, batch: pd.DataFrame):
        features = batch.drop(self.metrics, axis=1)
        non_drift_features = features.drop(self.drift_cols, axis=1)
        for metric in self.metrics:
            for i in range(self.n_models):
                self.results.iloc[
                    batch.index,
                    self.results.columns.get_loc(f"{metric}_pred_{i}_with_drift")] = \
                    self.meta_models[metric]["with_drift"][i].predict(features)
                self.results.iloc[
                    batch.index,
                    self.results.columns.get_loc(f"{metric}_pred_{i}_without_drift")] = \
                    self.meta_models[metric]["without_drift"][i].predict(non_drift_features)

    def _run_mtl(self):
        for index in range(0, self.metabase.shape[0] - self.train_batch_size, self.train_batch_size):
            train_batch = self.metabase.iloc[index:index + self.train_batch_size]
            self._train_metamodels(train_batch)

            pred_batch = self.metabase.iloc[index + self.train_batch_size:index + 2*self.train_batch_size]
            self._make_prediction(pred_batch)

    def _save_results(self):
        self.results.to_csv(f"results/results_dataframes/{self.output_filename}.csv", index=False)

        importances = self._get_importances()
        with open(f"results/results_importances/{self.output_filename}.json", "w") as fp:
            json.dump(importances, fp)

    def run(self):
        self._load_metabase()
        self._create_results_df()
        self._create_meta_models()
        self._get_drift_cols()
        self._run_mtl()
        self._save_results()

if __name__ == "__main__":
    d_gen = DriftContributionGenerator(
        base_model="SVC",
        dataset_name="rialto",
        train_batch_size=97,
        n_models=1,
    )
    d_gen.run()
    print("DONE")
