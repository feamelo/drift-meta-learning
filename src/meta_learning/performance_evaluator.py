import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, precision_score, recall_score


# Macros
CLF_METRICS_RANGE = {
    "precision": (0, 1),
    "recall": (0, 1),
    "f1-score": (0, 1),
    # "auc": (0, 1),
    "kappa": (0, 1),
}
REG_METRICS_RANGE = {
    "r2": (0, 1),
    "mse": (0, np.inf),
    "std": (0, np.inf),
}
METRICS = [*CLF_METRICS_RANGE.keys(), *REG_METRICS_RANGE.keys()]
METRICS_RANGE = {**CLF_METRICS_RANGE, **REG_METRICS_RANGE}


class PerformanceEvaluator():
    def __init__(self):
        self.binary_clf_metrics = CLF_METRICS_RANGE.keys()
        self.multiclass_clf_metrics = CLF_METRICS_RANGE.keys()
        self.reg_metrics = REG_METRICS_RANGE.keys()
        self.metrics_range = METRICS_RANGE
        self.metrics = METRICS

    def _get_auc(self, y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return auc(fpr, tpr)

    def _get_encoded(self, y_true, y_pred):
        """Encode categorical variables for avoiding errors
        on metric calculation"""
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit([*y_true, *y_pred])
        y_true = label_encoder.transform(y_true)
        y_pred = label_encoder.transform(y_pred)
        return y_true, y_pred

    def _get_performance(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        metric_name: str="precision",
    ) -> float:
        metric_dict = {
            "auc": self._get_auc,
            "kappa": cohen_kappa_score,
            "r2": r2_score,
            "mse": mean_squared_error,
            "std": lambda y_true, y_pred: np.std(y_true - y_pred),
            "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="micro"),
            "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="micro"),
            "f1-score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro"),
        }
        return metric_dict[metric_name](y_true, y_pred)

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, metric_name: str='precision') -> float:
        if metric_name not in METRICS:
            raise ValueError(f"'metric_name' param must be one of {self.metrics}")

        if metric_name in self.binary_clf_metrics:
            y_true, y_pred = self._get_encoded(y_true, y_pred)
        return self._get_performance(y_true, y_pred, metric_name)


evaluator = PerformanceEvaluator()
