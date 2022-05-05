import pandas as pd
from metrics import DomainClassifier, PsiCalculator, StatsMetrics
from meta_learning import PerformanceEvaluator


# Macros
BASE_MODEL_TYPE = 'classification'
BASE_MODEL_TYPES = ['classification', 'regression']
META_MODEL_HYPERPARAMETERS = {'num_leaves': 21, 'max_depth': 6}
META_LABEL_METRIC = 'precision'
LAST_SCORES_DELAY = 3
LAST_SCORES_N_SHIFTS = 3
OFFLINE_TRAIN_SPLIT = 0.5
R_STATE = 2022
OMEGA = 300  # Window size with known label
ETA = 100  # Window size with unlabeled examples
STEP = 10


performance = PerformanceEvaluator()


class MetaLearner():
    def __init__(
        self,
        base_model,
        meta_model,
        base_model_class_column:str,
        meta_label_metric: str=META_LABEL_METRIC,
        base_model_type: str=BASE_MODEL_TYPE,
        omega: int=OMEGA,
        eta: int=ETA,
        step: int=STEP,
        last_scores_delay: int=LAST_SCORES_DELAY,
        last_scores_n_shifts: int=LAST_SCORES_N_SHIFTS,
        offline_train_split: float=OFFLINE_TRAIN_SPLIT,
        ):
        self._get_performance_metrics(base_model_type, meta_label_metric)
        kwargs = locals()
        kwargs = {key: kwargs[key] for key in list(kwargs.keys()) if key not in ('self', '__class__')}
        self._update_params(**kwargs)

    def _update_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def _get_performance_metrics(self, base_model_type: str, meta_label_metric: str) -> None:
        if base_model_type not in BASE_MODEL_TYPES:
            raise Exception(f"Invalid base_model_type '{base_model_type}', must be one of: {BASE_MODEL_TYPES}")
        if meta_label_metric not in performance.clf_metrics:
            raise Exception(f"Invalid meta_label_metric '{meta_label_metric}', must be one of: {performance.clf_metrics}")

        if base_model_type == 'classification':
            self.performance_metrics = performance.clf_metrics
        else:
            self.performance_metrics = performance.reg_metrics

    def _fit_metrics(self, offline_df: pd.DataFrame) -> None:
        idx = int(self.offline_train_split * offline_df.shape[0])
        df = offline_df.iloc[:idx]
        X = df.drop(self.base_model_class_column, axis=1)

        self.fitted_metrics = [
            DomainClassifier().fit(X),
            PsiCalculator().fit(X),
            StatsMetrics().fit(X),
        ]

    def _get_last_performances(self, meta_base: pd.DataFrame) -> pd.DataFrame:
        df_cols = [col for col in meta_base.columns if '_t-' not in col]
        meta_base = meta_base[df_cols]

        for metric in self.performance_metrics:
            for n in range(self.last_scores_delay, self.last_scores_delay + self.last_scores_n_shifts):
                meta_base.loc[:, [f'{metric}_t-{n}']] = meta_base[metric].shift(n)
        return meta_base

    def _get_metafeatures(self, df: pd.DataFrame) -> pd.DataFrame:
        mf_dict = {}
        for metric in self.fitted_metrics:
            mf_dict = {**mf_dict, **metric.evaluate(df)}
        return pd.DataFrame.from_dict([mf_dict])

    def _get_offline_metabase(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = int(self.offline_train_split * df.shape[0])
        mf_df = df.iloc[idx:]
        meta_base = pd.DataFrame()
        offline_phase_size = mf_df.shape[0]
        upper_bound = offline_phase_size -  self.eta# - self.omega

        for t in range(0, upper_bound, self.step):
            # known_data = df.iloc[t:t+self.omega]
            arriving_data = mf_df.iloc[t:t + self.eta]

            # X_known = known_data.drop(self.base_model_class_column, axis=1)
            X_arriving = arriving_data.drop(self.base_model_class_column, axis=1)

            # meta features    
            mf = self._get_metafeatures(X_arriving)

            # meta label
            y_arriving = arriving_data[self.base_model_class_column] # no online só chega em t+1
            y_pred = self.base_model.predict(X_arriving)

            for metric in self.performance_metrics:
                mf[metric] = performance.evaluate(y_arriving, y_pred, metric)
            meta_base = pd.concat([meta_base, mf], ignore_index=True)
        meta_base = self._get_last_performances(meta_base)
        return meta_base

    def _train_base_model(self, offline_df: pd.DataFrame) -> None:
        idx = int(self.offline_train_split * offline_df.shape[0])
        df = offline_df.iloc[:idx]

        X = df.drop(self.base_model_class_column, axis=1)
        y = df[self.base_model_class_column]
        self.base_model.fit(X, y)

    def _train_meta_model(self) -> None:
        X = self.meta_base.drop(self.performance_metrics, axis=1)
        y = self.meta_base[self.meta_label_metric]
        self.meta_model.fit(X, y)

    def offline_stage(self, df: pd.DataFrame):
        self._train_base_model(df)
        self._fit_metrics(df)
        self.meta_base = self._get_offline_metabase(df)
        self._train_meta_model()
