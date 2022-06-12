import pandas as pd
from typing import Tuple
from sklearn.decomposition import PCA

# Custom imports
from metrics import DomainClassifier, PsiCalculator
from metrics import StatsMetrics, ClusteringMetrics
from meta_learning import evaluator


# Macros
BASE_MODEL_TYPE = 'classification'
BASE_MODEL_TYPES = ['classification', 'regression']
META_LABEL_METRIC = 'precision'
KNOWN_TARGET_DELAY = 300
KNOWN_TARGET_WINDOW_SIZE = 0
R_STATE = 2022
VERBOSE = False
ETA = 100  # Window size used to extract meta features
STEP = 10  # Step for next meta learning iteration
PCA_N_COMPONENTS = None  # Number of components to keep in dim reduction
# If < 1, select the num of components such that the amount of variance that
# needs to be explained is greater than the percentage specified by n_components.
DEFAULT_IMPUTE_VALUE = -9999


class MetaLearner():
    def __init__(
        self,
        base_model,
        meta_model,
        base_model_class_column:str,
        learning_window_size: int,
        meta_label_metric: str = META_LABEL_METRIC,
        base_model_type: str = BASE_MODEL_TYPE,
        eta: int = ETA,
        step: int = STEP,
        known_target_delay: int = KNOWN_TARGET_DELAY,
        known_target_window_size: int = KNOWN_TARGET_WINDOW_SIZE,
        pca_n_components: Tuple[int, float] = PCA_N_COMPONENTS,
        verbose: bool = VERBOSE, 
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
        if meta_label_metric not in evaluator.clf_metrics:
            raise Exception(f"Invalid meta_label_metric '{meta_label_metric}', must be one of: {evaluator.clf_metrics}")
        if base_model_type == 'classification':
            self.performance_metrics = evaluator.clf_metrics
        else:
            self.performance_metrics = evaluator.reg_metrics

    def _fit_metrics(self, offline_df: pd.DataFrame) -> None:
        df = offline_df.head(self.learning_window_size)
        X = df.drop(self.base_model_class_column, axis=1)
        X['predict_proba'] = self.base_model.predict_proba(X)[:, 0]

        self.fitted_metrics = [
            DomainClassifier().fit(X),
            PsiCalculator().fit(X),
            StatsMetrics().fit(X),
            ClusteringMetrics().fit(X),
        ]

    def _get_last_performances(self, meta_base: pd.DataFrame) -> pd.DataFrame:
        """Uses last known target window to measure old base model performance
        and use it as new meta features"""
        df_cols = [col for col in meta_base.columns if '_t-' not in col]
        meta_base = meta_base[df_cols]

        for metric in self.performance_metrics:
            for n in range(self.known_target_delay, self.known_target_delay + self.known_target_window_size):
                meta_base.loc[:, [f'{metric}_t-{n}']] = meta_base[metric].shift(n)
        return meta_base

    def _get_metafeatures(self, df: pd.DataFrame) -> pd.DataFrame:
        mf_dict = {}
        for metric in self.fitted_metrics:
            mf_dict = {**mf_dict, **metric.evaluate(df)}
        return pd.DataFrame.from_dict([mf_dict])

    def _get_offline_metabase(self, offline_df: pd.DataFrame) -> pd.DataFrame:
        mf_df = offline_df.tail(self.learning_window_size)
        meta_base = pd.DataFrame()
        offline_phase_size = mf_df.shape[0]
        upper_bound = offline_phase_size -  self.eta

        for t in range(0, upper_bound, self.step):
            arriving_data = mf_df.iloc[t:t + self.eta]
            X_arriving = arriving_data.drop(self.base_model_class_column, axis=1)
            y_pred = self.base_model.predict(X_arriving)
            X_arriving['predict_proba'] = self.base_model.predict_proba(X_arriving)[:, 0]

            # meta features    
            mf = self._get_metafeatures(X_arriving)

            # meta label - not available on online stage
            y_arriving = arriving_data[self.base_model_class_column]

            for metric in self.performance_metrics:
                mf[metric] = evaluator.evaluate(y_arriving, y_pred, metric)
            meta_base = pd.concat([meta_base, mf], ignore_index=True)
        meta_base = self._get_last_performances(meta_base)
        return meta_base

    def _train_base_model(self, offline_df: pd.DataFrame) -> None:
        df = offline_df.head(self.learning_window_size)

        X = df.drop(self.base_model_class_column, axis=1)
        y = df[self.base_model_class_column]
        self.base_model.fit(X, y)

    def _reduce_metabase(self, X: pd.DataFrame, stage: str="online") -> pd.DataFrame:
        if not self.pca_n_components:
            return X

        X = X.fillna(DEFAULT_IMPUTE_VALUE)
        if stage == "offline":
            svd_solver = "auto" if self.pca_n_components > 1 else "full"
            self.pca = PCA(
                n_components=self.pca_n_components,
                svd_solver=svd_solver,
                random_state=R_STATE
                ).fit(X)

        if self.verbose:
            n_comp = self.pca.n_components_
            variance = '{0:.2f}'.format(sum(self.pca.explained_variance_ratio_) * 100)
            print(f'Dim reduction - keeping {n_comp} components explaining {variance}% of variance')
        return pd.DataFrame(self.pca.transform(X))

    def _train_meta_model(self, stage: str="offline") -> None:
        X = self.meta_base.drop(self.performance_metrics, axis=1)
        y = self.meta_base[self.meta_label_metric]

        # Apply dimensionality reduction if set in init params
        X = self._reduce_metabase(X, stage)

        # Train meta model, if stage=online do incremental learning
        if stage == "offline":
            self.meta_model.fit(X, y)
        else:
            self.meta_model.partial_fit(X, y)

    def fit(self, df: pd.DataFrame) -> None:
        """Creates the first meta base and fit the first meta model"""
        self._train_base_model(df)
        self._fit_metrics(df)
        self.meta_base = self._get_offline_metabase(df)
        self._train_meta_model(stage="offline")

    def update(self, df: pd.DataFrame) -> None:
        """Update meta learner with new online data"""
        pass

    def target(self, y: Tuple[int, float, str]) -> None:
        """Update meta learner with upcoming target"""
        pass
