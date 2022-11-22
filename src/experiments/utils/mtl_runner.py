import time
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Custom classes
import sys
sys.path.insert(0,'..')
from meta_learning import MetaLearner
from utils import load_dataset

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')


# Macros
TIME_ELAPSED_FILE = "results/elapsed_time.csv"
R_STATE = 123
META_LABEL_METRIC = "kappa"
BASE_MODEL = RandomForestClassifier
DEFAULT_BASE_MODEL_PARAMS = {
    "verbose": True,
    "basis_model": RandomForestClassifier,
    "hyperparameters": {"max_depth": 6}
}


class MtLRunner():
    def __init__(
        self,
        dataset_name: str,
        class_col: str,
        offline_phase_size: int,
        base_train_size: int,
        target_delay: int,
        verbose: bool=True,
        include_drift_metrics: bool=True,
        base_model_params=DEFAULT_BASE_MODEL_PARAMS,
        meta_label_metric=META_LABEL_METRIC,
        save_file_name: str=None,
        **other_metalearner_params,
    ):
        self.verbose = verbose
        self.dataset_name = dataset_name
        self.class_col = class_col
        self.offline_phase_size = offline_phase_size
        self.base_train_size = base_train_size
        self.target_delay = target_delay
        self.include_drift_metrics = include_drift_metrics
        self.base_model_params = base_model_params
        self.meta_label_metric = meta_label_metric
        self.save_file_name = save_file_name
        self.learner = None
        self.dataset = load_dataset(dataset_name)
        self.other_metalearner_params = other_metalearner_params
        
        if not self.save_file_name:
            self.save_file_name = f"basemodel: {self.base_model_params['basis_model'].__name__} - dataset: {self.dataset_name} - with{'' if self.include_drift_metrics else 'out'}_drift_metrics"

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
        learner_params = {
            "base_model_params": self.base_model_params,
            "base_model_class_column": self.class_col,
            "verbose": True,
            "target_delay": self.target_delay,
            "include_drift_metrics_mfs": self.include_drift_metrics,
            **self.other_metalearner_params,
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
        # Save final metabase
        metabase = self.learner.metabase.metabase
        metabase.to_csv(f"metabases/{self.save_file_name}.csv", index=False)

        # Save final meta model
        for metric, metamodel in self.learner.meta_models.items():
            with open(f"models/{self.save_file_name} - metric: {metric}.pickle", "wb") as handle:
                pickle.dump(metamodel.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _print(self, msg: str):
        if self.verbose:
            print(msg)

    def run(self):
        """Run the MetaLearner for the specified data.
        Measures the elapsed time for running the meta learning algorithm
        and save it to a csv file for later comparison.
        """
        base_model_name = self.base_model_params["basis_model"].__name__

        print(f"Starting run -> {self.save_file_name}")
        # Load existing time elapsed dataframe
        time_df = pd.read_csv(TIME_ELAPSED_FILE)
        # Start timer
        start = time.time()

        # Run experiment
        self._print("Starting offline stage")
        self._run_offline_stage()
        self._print("Starting online stage")
        self._run_online_stage()
        self._print("Saving results")
        self._saving_results()

        # Append time elapsed data to dataframe
        time_df.loc[len(time_df)] = {
            "dataset": self.dataset_name,
            "base_model": base_model_name,
            "include_drift": self.include_drift_metrics,
            "total_elapsed_time": time.time() - start,
            **self.learner.elapsed_time
        }
        # Save execution data
        time_df.to_csv(TIME_ELAPSED_FILE, index=False)
