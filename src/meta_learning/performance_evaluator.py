import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score, mean_squared_error


# Macros
CLF_METRICS = ['precision', 'recall', 'f1-score', 'auc', 'kappa']
REG_METRICS = ['r2', 'mse', 'std']
METRICS = [*CLF_METRICS, *REG_METRICS]

class PerformanceEvaluator():
    def __init__(self):
        self.clf_metrics = CLF_METRICS
        self.reg_metrics = REG_METRICS
        self.metrics = METRICS
        pass

    def _get_auc(self, y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return auc(fpr, tpr)

    def _get_encoded(self, y_true, y_pred):
        le = preprocessing.LabelEncoder()
        le.fit(y_true)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        return y_true, y_pred

    def _get_performance(self, y_true, y_pred, metric_name='precision'):
        if metric_name == 'auc':
            return self._get_auc(y_true, y_pred)
        if metric_name == 'kappa':
            return cohen_kappa_score(y_true, y_pred)
        if metric_name == 'r2':
            return r2_score(y_true, y_pred)
        if metric_name == 'mse':
            return mean_squared_error(y_true, y_pred)
        if metric_name == 'std':
            return np.std(y_true - y_pred)

        label = list(set([*y_true, *y_pred]))[0]
        metrics = classification_report(y_true, y_pred, output_dict=True)[str(label)]
        return metrics[metric_name]

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, metric_name: str='precision') -> float:
        if metric_name not in METRICS:
            raise ValueError(f"'metric_name' param must be one of {METRICS}")

        if metric_name in CLF_METRICS:
            y_true, y_pred = self._get_encoded(y_true, y_pred)
        return self._get_performance(y_true, y_pred, metric_name)