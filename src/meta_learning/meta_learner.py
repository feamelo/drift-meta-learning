import pandas as pd
from typing import Tuple

# Custom imports
from metrics import DomainClassifier, PsiCalculator
from metrics import StatsMetrics, ClusteringMetrics
from meta_learning import evaluator, Metabase, BaseLevelBase


# Macros
PREDICTION_COL = 'predict'
META_PREDICTION_COL = 'predicted'
BASE_MODEL_TYPE = 'classification'
BASE_MODEL_TYPES = ['classification', 'regression']
META_LABEL_METRIC = 'precision'
R_STATE = 2022
VERBOSE = False
ETA = 100  # Window size used to extract meta features
STEP = 10  # Step for next meta learning iteration
PCA_N_COMPONENTS = None  # Number of components to keep in dim reduction
# If < 1, select the num of components such that the amount of variance that
# needs to be explained is greater than the percentage specified by n_components.


class MetaLearner():
    def __init__(
        self,
        base_model,
        meta_model,
        base_model_class_column:str,
        meta_label_metric: str = META_LABEL_METRIC,
        base_model_type: str = BASE_MODEL_TYPE,
        eta: int = ETA,
        step: int = STEP,
        pca_n_components: Tuple[int, float] = PCA_N_COMPONENTS,
        verbose: bool = VERBOSE, 
        ):
        self.performance_metrics = self._get_performance_metrics(base_model_type, meta_label_metric)
        self.__dict__.update(locals())
        
        self.prediction_col = PREDICTION_COL

        self.metabase = Metabase(
            pca_n_components = pca_n_components,
            target_col = meta_label_metric,
            prediction_col = META_PREDICTION_COL,
            verbose = verbose,
        )

        self.baselevel_base = BaseLevelBase(
            batch_size = eta,
            target_col = base_model_class_column,
            prediction_col = self.prediction_col,
            verbose = verbose,
        )

    def _get_performance_metrics(self, base_model_type: str, meta_label_metric: str) -> None:
        if base_model_type not in BASE_MODEL_TYPES:
            raise Exception(f"Invalid base_model_type '{base_model_type}', must be one of: {BASE_MODEL_TYPES}")
        if meta_label_metric not in evaluator.clf_metrics:
            raise Exception(f"Invalid meta_label_metric '{meta_label_metric}', must be one of: {evaluator.clf_metrics}")
        if base_model_type == 'classification':
            return evaluator.clf_metrics
        else:
            return evaluator.reg_metrics

    def _fit_metrics(self, train_df: pd.DataFrame) -> None:
        X = train_df.drop(self.base_model_class_column, axis=1)
        X['predict_proba'] = self.base_model.predict_proba(X)[:, 0]
        self.fitted_metrics = [
            DomainClassifier().fit(X),
            PsiCalculator().fit(X),
            StatsMetrics().fit(X),
            ClusteringMetrics().fit(X),
        ]

    def _get_meta_features(self, batch_features: pd.DataFrame) -> pd.DataFrame:
        mf_dict = {}
        for metric in self.fitted_metrics:
            mf_dict = {**mf_dict, **metric.evaluate(batch_features)}
        return pd.DataFrame.from_dict([mf_dict])

    def _get_meta_labels(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        # Not available on online stage
        y_true = df_batch[self.base_model_class_column]
        y_pred = df_batch[self.prediction_col]
        return evaluator.evaluate(y_true, y_pred, self.meta_label_metric)

    def _fit_offline_baselevel_base(self, dataframe: pd.DataFrame) -> None:
        # create prediction and predict_proba columns
        features = dataframe.drop(self.base_model_class_column, axis=1)
        dataframe["predict_proba"] = self.base_model.predict_proba(features)[:, 0]
        dataframe[self.prediction_col] = self.base_model.predict(features)
        self.baselevel_base.fit(dataframe)

    def _fit_offline_metabase(self) -> pd.DataFrame:
        meta_base = pd.DataFrame()
        offline_base = self.baselevel_base.get_raw()
        offline_phase_size = offline_base.shape[0]
        upper_bound = offline_phase_size -  self.eta

        for t in range(0, upper_bound, self.step):
            df_batch = offline_base.iloc[t:t + self.eta]
            batch_features = df_batch.drop([self.base_model_class_column, self.prediction_col], axis=1)
            mf =  self._get_meta_features(batch_features)
            mf[self.meta_label_metric] = self._get_meta_labels(df_batch)

            meta_base = pd.concat([meta_base, mf], ignore_index=True)
        self.metabase.fit(meta_base)

    def _train_base_model(self, train_df: pd.DataFrame) -> None:
        X = train_df.drop(self.base_model_class_column, axis=1)
        y = train_df[self.base_model_class_column]
        self.base_model.fit(X, y)

    def _get_train_metabase(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.metabase.get_train_metabase()
        X = df.drop(self.meta_label_metric, axis=1)
        y = df[self.meta_label_metric]
        return X, y

    def _train_meta_model(self) -> None:
        X, y = self._get_train_metabase()

        if self.verbose:
            print("Training meta model")

        # Incremental train of meta model
        self.meta_model.partial_fit(X, y)

    def _check_drift(self) -> None:
        # TO DO
        pass

    def fit(self, base_train_df: pd.DataFrame, meta_train_df: pd.DataFrame) -> None:
        """Creates the first meta base and fits the first meta model"""
        self._train_base_model(base_train_df.copy())
        self._fit_metrics(base_train_df.copy())
        self._fit_offline_baselevel_base(meta_train_df.copy())
        self._fit_offline_metabase()
        self._train_meta_model()

        # Update metabase with meta model prediction
        X, _ = self._get_train_metabase()
        y_pred = self.meta_model.predict(X)
        self.metabase.update_predictions(y_pred)
        return self

    def update(self, new_instance: pd.DataFrame) -> None:
        """Update meta learner with new online data"""
        # Update base level base
        new_instance = pd.DataFrame(new_instance).T
        dataframe = new_instance.copy()
        dataframe["predict_proba"] = self.base_model.predict_proba(new_instance)[:, 0]
        dataframe[self.prediction_col] = self.base_model.predict(new_instance)
        self.baselevel_base.update(dataframe)

        # If there is a new batch for calculating meta fetures
        if self.baselevel_base.new_batch_counter == self.step:
            batch = self.baselevel_base.get_batch()
            batch_features = batch.drop(self.prediction_col, axis=1)
            mf = self._get_meta_features(batch_features)
            mf[self.metabase.prediction_col] = self.meta_model.predict(mf)
            self.metabase.update(mf)
            
            # Check if the new batch is a drift indicative
            self._check_drift()

    def update_target(self, y: Tuple[int, float, str]) -> None:
        """Update meta learner with upcoming target"""
        self.baselevel_base.update_target(y)

        # If there is a new batch for calculating meta labels
        if self.baselevel_base.new_target_batch_counter == self.step:
            batch = self.baselevel_base.get_target_batch()
            meta_label = self._get_meta_labels(batch)
            self.metabase.update_target(meta_label)

            if self.metabase.new_batch_size == self.metabase.learning_window_size:
                self._train_meta_model()  # Incremental learning