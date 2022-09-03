import pickle
from sklearn.ensemble import RandomForestClassifier

# Custom classes
import sys
sys.path.insert(0,'..')
from meta_learning import MetaLearner, Model, MetaModel
from utils import load_dataset

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')


# Macros
R_STATE = 123
META_LABEL_METRIC = "kappa"
BASE_MODEL = RandomForestClassifier
DEFAULT_BASE_MODEL = Model(
    verbose=True,
    basis_model=RandomForestClassifier,
    hyperparameters={"max_depth": 6}
)


class MetabaseGenerator():
    def __init__(
        self,
        dataset_name: str,
        class_col: str,
        base_model_type: str,
        offline_phase_size: int,
        base_train_size: int,
        eta: int,
        step: int,
        target_delay: int,
        verbose: bool=True,
        include_drift_metrics: bool=True,
        base_model=DEFAULT_BASE_MODEL,
        meta_label_metric=META_LABEL_METRIC,
    ):
        self.verbose = verbose
        self.dataset_name = dataset_name
        self.class_col = class_col
        self.base_model_type = base_model_type
        self.offline_phase_size = offline_phase_size
        self.base_train_size = base_train_size
        self.eta = eta
        self.step = step
        self.target_delay = target_delay
        self.include_drift_metrics = include_drift_metrics
        self.base_model = base_model
        self.meta_label_metric = meta_label_metric
        self.learner = None
        self.dataset = load_dataset(dataset_name)

    def _get_offline_data(self):
        offline_df = self.dataset.iloc[:self.offline_phase_size]
        base_train_df = offline_df.iloc[:self.base_train_size]
        meta_train_df = offline_df.iloc[self.base_train_size:]
        return base_train_df, meta_train_df

    def _get_online_data(self):
        online_df = self.dataset.iloc[self.offline_phase_size:]
        online_features = online_df.drop(self.class_col, axis=1).reset_index(drop=True)
        online_targets = online_df[self.class_col].reset_index(drop=True)
        return online_features, online_targets

    def _run_offline_stage(self):
        meta_model = MetaModel()
        learner_params = {
            "base_model": self.base_model,
            "meta_model": meta_model,
            "base_model_class_column": self.class_col,
            "eta": self.eta,
            "step": self.step,
            "meta_label_metric": self.meta_label_metric,
            "verbose": True,
            "target_delay": self.target_delay,
            "base_model_type": self.base_model_type,
            "include_drift_metrics_mfs": self.include_drift_metrics,
            }
        base_train_df, meta_train_df = self._get_offline_data()
        self.learner = MetaLearner(**learner_params).fit(base_train_df, meta_train_df)

    def _run_online_stage(self):
        online_features, online_targets = self._get_online_data()

        # start - no target
        for _, row in online_features.iloc[:self.target_delay].iterrows():
            self.learner.update(row)

        # middle - both target and instances
        df = online_features.iloc[self.target_delay:-self.target_delay]
        for i, row in df.iterrows():
            self.learner.update(row)
            self.learner.update_target(online_targets.iloc[i - self.target_delay])

        # end - only targets
        for target in online_targets.tail(self.target_delay):
            self.learner.update_target(target)

    def _saving_results(self):
        file_name = f"basemodel: {BASE_MODEL.__name__} - metric: {META_LABEL_METRIC} - dataset: {self.dataset_name}"
        if self.include_drift_metrics:
            file_name += " - with_drift_metrics"

        # Save final metabase
        metabase = self.learner.metabase.metabase
        metabase.to_csv(f"metabases/{file_name}.csv", index=False)

        # Save final meta model
        metamodel = self.learner.meta_model.model
        with open(f"models/{file_name}.pickle", "wb") as handle:
            pickle.dump(metamodel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _print(self, msg: str):
        if self.verbose:
            print(msg)

    def run(self):
        self._print("Starting offline stage")
        self._run_offline_stage()
        self._print("Starting online stage")
        self._run_online_stage()
        self._print("Saving results")
        self._saving_results()
