import pandas as pd
import time
from typing import Tuple
from threading import Thread

# Custom imports
from metrics import PsiCalculator, Udetector, DomainClassifier, OmvPht
from metrics import StatsMetrics, ClusteringMetrics, SqsiCalculator
from meta_learning import evaluator, Metabase, BaseLevelBase
from meta_learning import BaseModel, MetaModel



# Macros
BASE_PREDICTION_COL = "predict"
META_PREDICTION_COL = "predicted_"
BASE_MODEL_TYPE = "binary_classification"
BASE_MODEL_TYPES = ["binary_classification", "multiclass", "regression"]
META_LABEL_METRIC = "kappa"
BASELINE_COL_SUFFIX = "last_"
R_STATE = 2022
VERBOSE = False
ETA = 300  # Window size used to extract meta features
STEP = 30  # Step for next meta learning iteration
PCA_N_COMPONENTS = None  # Number of components to keep in dim reduction
# If < 1, select the num of components such that the amount of variance that
# needs to be explained is greater than the percentage specified by n_components.
INCLUDE_DRIFT_METRICS_MFS = True  # Indicates if drift detection techniques should
# be used as meta features


class MetaLearner():
    """Class that implements the meta learning algorithm.

    Args:
        base_model:
            An initialized instance of sklearn/lgbm model. It needs to have the methods
            .fit, .predict and .predict_proba
            For hyperparam optimization, use the BaseModel wrapper class of meta_learning folder.
        meta_model:
            An initialized instance of sklearn model. It needs to have the methods
            .fit, .predict and .predict_proba. For lgbm with optuna hyperparam optimization, use
            the MetaModel wrapper class of meta_learning folder.
        base_model_class_column (str):
            Column name containing the target variable
        target_delay (int):
            Number of instances for arriving the target
        meta_label_metric (str, optional):
            string containing the metric to be used on base model evaluation. It can be:
            - regression: mse/r2
            - classification: AUC/kappa/recall/precision/F1/accuracy
            Defaults to "kappa"
        base_model_type (str, optional):
            string containing the base model type name between
            ["binary_classification", "multiclass", "regression"]
            Defaults to "binary_classification"
        eta (int, optional):
            Window size (number of instances) to calculate the meta features. Defaults to 300.
        step (int, optional):
            Step (number of instances) for next meta learning iteration. Defaults to 30.
        pca_n_components (Tuple[int, float], optional):
            Number of components to keep in metabase dimensionality reduction.
            If < 1, select the num of components such that the amount of variance that
            needs to be explained is greater than the percentage specified by n_components.
            If None, no PCA will by applied. Defaults to None.
        verbose (bool, optional):
            verbosity. Defaults to False.
        include_drift_metrics_mfs (bool, optional):
            Boolean indicating wether the drift metrics should be used as meta features
            Defaults to True.
    """
    def __init__(
        self,
        base_model_class_column:str,
        target_delay: int,
        base_model_params: dict={},
        meta_model_params: dict={},
        meta_label_metrics: list = [],
        base_model_type: str = BASE_MODEL_TYPE,
        eta: int = ETA,
        step: int = STEP,
        pca_n_components: Tuple[int, float] = PCA_N_COMPONENTS,
        verbose: bool = VERBOSE,
        include_drift_metrics_mfs: bool = INCLUDE_DRIFT_METRICS_MFS,
    ):
        self.base_prediction_col = BASE_PREDICTION_COL
        self.fitted_metrics = []
        self.base_model = BaseModel(**base_model_params)
        self.target_delay = target_delay
        self.base_model_class_column = base_model_class_column
        self.base_model_type = base_model_type
        self.eta = eta
        self.step = step
        self.pca_n_components = pca_n_components
        self.verbose = verbose
        self.include_drift_metrics_mfs = include_drift_metrics_mfs
        self.meta_label_metrics = meta_label_metrics
        self.elapsed_time = {}

        self.performance_metrics = self._get_performance_metrics(base_model_type, meta_label_metrics)
        if not meta_label_metrics:
            self.meta_label_metrics = self.performance_metrics
        self.meta_models = {metric: MetaModel(**meta_model_params) for metric in self.meta_label_metrics}

        self.metabase = Metabase(
            pca_n_components = pca_n_components,
            prediction_col_suffix = META_PREDICTION_COL,
            verbose = verbose,
        )
        self.baselevel_base = BaseLevelBase(
            batch_size = eta,
            target_col = base_model_class_column,
            prediction_col = self.base_prediction_col,
            verbose = verbose,
        )

    def _get_performance_metrics(self, base_model_type: str, meta_label_metrics: list) -> list:
        """Check if the chosen metric can be used for evaluating the base model type.

        Args:
            base_model_type (str): string containing the base model type name
                between ["binary_classification", "multiclass", "regression"]
            meta_label_metric (str): string containing the metric to be used on
                base model evaluation. It can be:
                - regression: mse/r2
                - classification: AUC/kappa/recall/precision/F1/accuracy

        Returns:
            list: List of metrics that can be used for base model type
        """
        if base_model_type not in BASE_MODEL_TYPES:
            raise Exception(f"Invalid base_model_type '{base_model_type}', \
                must be one of: {BASE_MODEL_TYPES}")

        metric_dict = {
            "binary_classification": evaluator.binary_clf_metrics,
            "multiclass": evaluator.multiclass_clf_metrics,
            "regression": evaluator.reg_metrics,
        }
        metrics = metric_dict[base_model_type]

        for meta_label_metric in meta_label_metrics:
            if meta_label_metric not in metrics:
                raise Exception(f"Invalid meta_label_metric '{meta_label_metric}' \
                    for model type {base_model_type}, must be one of: {metrics}")
        return metrics

    def _fit_drift_metrics(self, train_df: pd.DataFrame) -> None:
        """Fit drift metrics with reference (no drifted) data
        used for base model training.
        """
        features = train_df.rename(columns={
            self.base_model_class_column: self.base_prediction_col})
        pred_proba = self.base_model.predict_proba(features.drop(self.base_prediction_col, axis=1))
        score_cols = []
        for idx, pred in enumerate(pred_proba.T):
            features[f"predict_proba_{idx}"] = pred
            score_cols.append(f"predict_proba_{idx}")
        self.fitted_metrics = [
            StatsMetrics().fit(features),
            ClusteringMetrics().fit(features),
        ]
        if self.include_drift_metrics_mfs:
            self.fitted_metrics += [
                PsiCalculator().fit(features),
                DomainClassifier().fit(features),
                OmvPht(score_cols=score_cols).fit(features),
                SqsiCalculator(score_cols=score_cols).fit(features),
                Udetector(prediction_col=self.base_prediction_col).fit(features),
            ]
        
        for metric in self.fitted_metrics:
            self.elapsed_time[metric.__class__.__name__] = 0

    def _get_meta_features(self, batch_features: pd.DataFrame) -> pd.DataFrame:
        """Calculates the meta features by evaluating all metrics of the
        self.fitted_metrics array. Usage of threading for faster return.

        Args:
            batch_features (pd.DataFrame): batch of features to calculate the
            metrics on.

        Returns:
            pd.DataFrame: single row dataframe with all evaluated metrics
            for the provided batch
        """
        mf_dict = dict()
        def calculate_mf(metric, batch_features):
            start = time.time()
            mf_dict.update(metric.evaluate(batch_features))
            self.elapsed_time[metric.__class__.__name__] += time.time() - start
            
        threads = list()
        for metric in self.fitted_metrics:
            thread = Thread(
                target=calculate_mf,
                args=(metric, batch_features)
            )
            threads.append(thread)

        # start threads
        for thread in threads:
            thread.start()
        # wait for threads to finish
        for thread in threads:
            thread.join()
        return pd.DataFrame.from_dict([mf_dict])

    def _limit_metric_value(self, value: float, metric_name: str) -> float:
        """Apply a limit to the metric value (considering its theoretical
        knowledge) when the regression predicted value is above that limit.
        e.g. precision outside the [0, 1] limits

        Args:
            value (float): predicted value
            metric_name (str): metric name

        Returns:
            float: limited value
        """
        if value <= evaluator.metrics_range[metric_name][0]:
            return evaluator.metrics_range[metric_name][0]
        if value >= evaluator.metrics_range[metric_name][1]:
            return evaluator.metrics_range[metric_name][1]
        return value

    def _get_meta_labels(self, df_batch: pd.DataFrame) -> dict:
        """Calculate the meta labels (base model performance)
        for the given batch"""
        # Not available on online stage
        y_true = df_batch[self.base_model_class_column]
        y_pred = df_batch[self.base_prediction_col]
        metrics = {}
        for metric_name in self.performance_metrics:
            metrics[metric_name] = evaluator.evaluate(y_true, y_pred, metric_name)
            # metric_value = evaluator.evaluate(y_true, y_pred, metric_name) #@@
            # metrics[metric_name] = self._limit_metric_value(metric_value, metric_name) #@@
        return metrics

    def _fit_offline_baselevel_base(self, dataframe: pd.DataFrame) -> None:
        """Save offline dataset (for base level, used for base model training)
        and creates prediction and predict_proba columns for later use"""
        # create prediction and predict_proba columns
        features = dataframe.drop(self.base_model_class_column, axis=1)
        pred_proba = self.base_model.predict_proba(features)
        for idx, pred in enumerate(pred_proba.T):
            dataframe[f"predict_proba_{idx}"] = pred
        dataframe[self.base_prediction_col] = self.base_model.predict(features)
        self.baselevel_base.fit(dataframe)

    def _get_last_performances(self, meta_base: pd.DataFrame) -> pd.DataFrame:
        """The baseline is the last calculated performance (for the last
        batch with known target). Get the last performances of a batch
        for offline stage usage.
        """
        for metric in self.performance_metrics:
            col_name = f"{BASELINE_COL_SUFFIX}{metric}"
            meta_base.loc[:, col_name] = meta_base[metric].shift(self.target_delay)
        return meta_base

    def _fit_offline_metabase(self) -> None:
        """Splits the offline database in groups of fixed size (self.eta)
        and iterate over them by steps of fixed size (self.step).
        Calculate the meta features and meta labels for each group and
        concat them to create the first metabase.
        """
        meta_base = pd.DataFrame()
        offline_base = self.baselevel_base.get_raw()
        offline_phase_size = offline_base.shape[0]
        upper_bound = offline_phase_size -  self.eta

        for time in range(0, upper_bound, self.step):
            df_batch = offline_base.iloc[time:time + self.eta]
            batch_features = df_batch.drop([self.base_model_class_column], axis=1)
            meta_features =  self._get_meta_features(batch_features)
            meta_labels = self._get_meta_labels(df_batch)
            meta_features[list(meta_labels.keys())] = list(meta_labels.values())

            meta_base = pd.concat([meta_base, meta_features], ignore_index=True)
        meta_base = self._get_last_performances(meta_base)
        self.metabase.fit(meta_base)

    def _train_base_model(self, train_df: pd.DataFrame) -> None:
        features = train_df.drop(self.base_model_class_column, axis=1)
        target = train_df[self.base_model_class_column]
        self.base_model.fit(features, target)

    def _get_train_metabase(self, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        meta_base = self.metabase.get_train_metabase()
        features = meta_base.drop(self.performance_metrics, axis=1)
        target = meta_base[target_col]
        return features, target

    def _train_meta_model(self) -> None:
        """Train one meta model for each performance metric
        listed on self.meta_label_metrics"""
        for metric in self.meta_label_metrics:
            features, target = self._get_train_metabase(metric)
            self.meta_models[metric].fit(features, target)

    def _get_baseline(self) -> dict:
        """The baseline is the last calculated performance (for the last
        batch with known target). Get the last performances of a single
        instance for online stage usage.

        Returns:
            dict: dictionary containing the last performance for each of the
            possible metrics, a baseline suffix is added on key names.
        """
        batch = self.metabase.get_last_performed_batch()[self.performance_metrics]
        return {f"{BASELINE_COL_SUFFIX}{metric}": value for metric, value in batch.iteritems()}

    def fit(self, base_train_df: pd.DataFrame, meta_train_df: pd.DataFrame) -> None:
        """Creates the first meta base and fits the first meta model"""
        self._train_base_model(base_train_df.copy())
        self._fit_drift_metrics(base_train_df.copy())
        self._fit_offline_baselevel_base(meta_train_df.copy())
        self._fit_offline_metabase()
        self._train_meta_model()

        # Update metabase with meta model prediction
        features, _ = self._get_train_metabase(self.meta_label_metrics[0])
        for metric in self.meta_label_metrics:
            y_pred = self.meta_models[metric].predict(features)
            self.metabase.update_predictions(
                prediction=y_pred,
                prediction_col=f"{META_PREDICTION_COL}{metric}")
        return self

    def update(self, new_instance: pd.DataFrame) -> None:
        """Update meta learner with new online data"""
        # Update base level base
        new_instance_df = pd.DataFrame(new_instance).T

        # Create multiple cols for predict proba output
        pred_proba = self.base_model.predict_proba(new_instance_df)
        for idx, pred in enumerate(pred_proba.T):
            new_instance[f"predict_proba_{idx}"] = pred[0]
        new_instance[self.base_prediction_col] = self.base_model.predict(new_instance_df)[0]
        self.baselevel_base.update(new_instance)

        # If there is a new batch for calculating meta features
        if self.baselevel_base.new_batch_counter == self.step:
            baseline = self._get_baseline()
            batch = self.baselevel_base.get_batch()
            meta_features = self._get_meta_features(batch)
            meta_features[list(baseline.keys())] = list(baseline.values())

            for metric in self.meta_label_metrics:
                predicted = pd.Series(self.meta_models[metric].predict(meta_features))
                predicted = predicted.apply(lambda x: self._limit_metric_value(x, metric))
                meta_features[f"{META_PREDICTION_COL}{metric}"] = predicted
            self.metabase.update(meta_features)

    def update_target(self, target: Tuple[int, float, str]) -> None:
        """Update meta learner with upcoming target"""
        self.baselevel_base.update_target(target)

        # If there is a new batch for calculating meta labels
        if self.baselevel_base.new_target_batch_counter == self.step:
            batch = self.baselevel_base.get_target_batch()
            meta_labels = self._get_meta_labels(batch)
            self.metabase.update_target(meta_labels)

            if self.metabase.new_batch_size == self.step:
                self._train_meta_model()
