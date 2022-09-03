import pandas as pd
from typing import Tuple
from threading import Thread

# Custom imports
from metrics import PsiCalculator, Udetector, DomainClassifier, OmvPht
from metrics import StatsMetrics, ClusteringMetrics, SqsiCalculator
from meta_learning import evaluator, Metabase, BaseLevelBase


# Macros
PREDICTION_COL = "predict"
META_PREDICTION_COL = "predicted"
BASE_MODEL_TYPE = "binary_classification"
BASE_MODEL_TYPES = ["binary_classification", "multiclass", "regression"]
META_LABEL_METRIC = "precision"
BASELINE_COL_SUFFIX = "last_"
R_STATE = 2022
VERBOSE = False
ETA = 100  # Window size used to extract meta features
STEP = 10  # Step for next meta learning iteration
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
        meta_model: An initialized instance of sklearn model.
        base_model_class_column (str): _description_
        target_delay (int): _description_
        meta_label_metric (str, optional): _description_. Defaults to META_LABEL_METRIC.
        base_model_type (str, optional): _description_. Defaults to BASE_MODEL_TYPE.
        eta (int, optional): _description_. Defaults to ETA.
        step (int, optional): _description_. Defaults to STEP.
        pca_n_components (Tuple[int, float], optional): _description_. Defaults to PCA_N_COMPONENTS.
        verbose (bool, optional): _description_. Defaults to VERBOSE.
        include_drift_metrics_mfs (bool, optional): _description_. Defaults to INCLUDE_DRIFT_METRICS_MFS.
    """
    def __init__(
        self,
        base_model,
        meta_model,
        base_model_class_column:str,
        target_delay: int,
        meta_label_metric: str = META_LABEL_METRIC,
        base_model_type: str = BASE_MODEL_TYPE,
        eta: int = ETA,
        step: int = STEP,
        pca_n_components: Tuple[int, float] = PCA_N_COMPONENTS,
        verbose: bool = VERBOSE,
        include_drift_metrics_mfs: bool = INCLUDE_DRIFT_METRICS_MFS,
        ):
        self.prediction_col = PREDICTION_COL
        self.fitted_metrics = []
        self.base_model = base_model
        self.meta_model = meta_model
        self.target_delay = target_delay
        self.base_model_class_column = base_model_class_column
        self.meta_label_metric = meta_label_metric
        self.base_model_type = base_model_type
        self.eta = eta
        self.step = step
        self.pca_n_components = pca_n_components
        self.verbose = verbose
        self.include_drift_metrics_mfs = include_drift_metrics_mfs

        self.performance_metrics = self._get_performance_metrics(base_model_type, meta_label_metric)
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

    def _get_performance_metrics(self, base_model_type: str, meta_label_metric: str) -> list:
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

        if meta_label_metric not in metrics:
            raise Exception(f"Invalid meta_label_metric '{meta_label_metric}' \
                for model type {base_model_type}, must be one of: {metrics}")
        return metrics

    def _fit_metrics(self, train_df: pd.DataFrame) -> None:
        features = train_df.rename(columns={
            self.base_model_class_column: self.prediction_col})
        pred_proba = self.base_model.predict_proba(features.drop(self.prediction_col, axis=1))
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
                Udetector(prediction_col=self.prediction_col).fit(features),
            ]

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
        threads = list()
        for metric in self.fitted_metrics:
            thread = Thread(
                target=lambda metric, batch_features, mf_dict: mf_dict.update(metric.evaluate(batch_features)),
                args=(metric, batch_features, mf_dict)
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
        if value <= evaluator.metrics_range[metric_name][0]:
            return evaluator.metrics_range[metric_name][0]
        if value >= evaluator.metrics_range[metric_name][1]:
            return evaluator.metrics_range[metric_name][1]
        return value

    def _get_meta_labels(self, df_batch: pd.DataFrame) -> dict:
        # Not available on online stage
        y_true = df_batch[self.base_model_class_column]
        y_pred = df_batch[self.prediction_col]
        metrics = {}
        for metric_name in self.performance_metrics:
            metric_value = evaluator.evaluate(y_true, y_pred, metric_name)
            metrics[metric_name] = self._limit_metric_value(metric_value, metric_name)
        return metrics

    def _fit_offline_baselevel_base(self, dataframe: pd.DataFrame) -> None:
        # create prediction and predict_proba columns
        features = dataframe.drop(self.base_model_class_column, axis=1)
        pred_proba = self.base_model.predict_proba(features)
        for idx, pred in enumerate(pred_proba.T):
            dataframe[f"predict_proba_{idx}"] = pred
        dataframe[self.prediction_col] = self.base_model.predict(features)
        self.baselevel_base.fit(dataframe)

    def _get_last_performances(self, meta_base: pd.DataFrame) -> pd.DataFrame:
        for metric in self.performance_metrics:
            col_name = f"{BASELINE_COL_SUFFIX}{metric}"
            meta_base.loc[:, col_name] = meta_base[metric].shift(self.target_delay)
        return meta_base

    def _fit_offline_metabase(self) -> pd.DataFrame:
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

    def _get_train_metabase(self) -> Tuple[pd.DataFrame, pd.Series]:
        meta_base = self.metabase.get_train_metabase()
        features = meta_base.drop(self.performance_metrics, axis=1)
        target = meta_base[self.meta_label_metric]
        return features, target

    def _train_meta_model(self) -> None:
        features, target = self._get_train_metabase()
        self.meta_model.fit(features, target)

    def _check_drift(self) -> None:
        # TODO
        pass

    def _get_baseline(self) -> dict:
        """The baseline is the last calculated performance (for the last
        batch with known target).

        Returns:
            dict: dictionary containing the last performance for each of the
            possible metrics, a baseline suffix is added on key names.
        """
        batch = self.metabase.get_last_performed_batch()[self.performance_metrics]
        return {f"{BASELINE_COL_SUFFIX}{metric}": value for metric, value in batch.iteritems()}

    def fit(self, base_train_df: pd.DataFrame, meta_train_df: pd.DataFrame) -> None:
        """Creates the first meta base and fits the first meta model"""
        self._train_base_model(base_train_df.copy())
        self._fit_metrics(base_train_df.copy())
        self._fit_offline_baselevel_base(meta_train_df.copy())
        self._fit_offline_metabase()
        self._train_meta_model()

        # Update metabase with meta model prediction
        features, _ = self._get_train_metabase()
        y_pred = self.meta_model.predict(features)
        self.metabase.update_predictions(y_pred)
        return self

    def update(self, new_instance: pd.DataFrame) -> None:
        """Update meta learner with new online data"""
        # Update base level base
        new_instance_df = pd.DataFrame(new_instance).T

        # Create multiple cols for predict proba output
        pred_proba = self.base_model.predict_proba(new_instance_df)
        for idx, pred in enumerate(pred_proba.T):
            new_instance[f"predict_proba_{idx}"] = pred[0]
        new_instance[self.prediction_col] = self.base_model.predict(new_instance_df)[0]
        self.baselevel_base.update(new_instance)

        # If there is a new batch for calculating meta fetures
        if self.baselevel_base.new_batch_counter == self.step:
            baseline = self._get_baseline()
            batch = self.baselevel_base.get_batch()
            meta_features = self._get_meta_features(batch)
            meta_features[list(baseline.keys())] = list(baseline.values())
            meta_features[self.metabase.prediction_col] = self.meta_model.predict(meta_features)
            self.metabase.update(meta_features)

            # Check if the new batch is a drift indicative
            self._check_drift()

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
