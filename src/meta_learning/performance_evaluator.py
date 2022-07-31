import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score, mean_squared_error


# Macros
CLF_METRICS_RANGE = {
    "precision": (0, 1),
    "recall": (0, 1),
    "f1-score": (0, 1),
    "auc": (0, 1),
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
        self.clf_metrics = CLF_METRICS_RANGE.keys()
        self.reg_metrics = REG_METRICS_RANGE.keys()
        self.metrics_range = METRICS_RANGE
        self.metrics = METRICS

    def _get_auc(self, y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return auc(fpr, tpr)

    def _get_encoded(self, y_true, y_pred):
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(y_true)
        y_true = label_encoder.transform(y_true)
        y_pred = label_encoder.transform(y_pred)
        return y_true, y_pred

    def _get_performance(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        metric_name: str='precision',
    ) -> float:
        metric_dict = {
            "auc": self._get_auc,
            "kappa": cohen_kappa_score,
            "r2": r2_score,
            "mse": mean_squared_error,
            "std": lambda y_true, y_pred: np.std(y_true - y_pred),
        }
        if metric_name in metric_dict:
            return metric_dict[metric_name](y_true, y_pred)

        # get metrics from classification report
        label = list(set([*y_true, *y_pred]))[0]
        metrics = classification_report(y_true, y_pred, output_dict=True)[str(label)]
        return metrics[metric_name]

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, metric_name: str='precision') -> float:
        if metric_name not in METRICS:
            raise ValueError(f"'metric_name' param must be one of {self.metrics}")

        if metric_name in self.clf_metrics:
            y_true, y_pred = self._get_encoded(y_true, y_pred)
        return self._get_performance(y_true, y_pred, metric_name)

    def box_plot(
        self,
        results_df: pd.DataFrame,
        y_true_col: str = None,
        cols_to_plot: list = [],
        title: str = "Box plot",
        figsize = (15, 10),
    ) -> None:
        props = dict(boxes="LightBlue", whiskers="DarkOrange", medians="DarkBlue", caps="Gray")

        if y_true_col:
            for col in cols_to_plot:
                results_df[col] = results_df[y_true_col] - results_df[col]
            results_df = results_df[cols_to_plot]
        results_df.plot.box(
            color = props,
            patch_artist = True,
            fontsize = 20,
            title = title,
            figsize = figsize
        )

    def cumulative_gain(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_baseline: pd.Series,
        title: str = "Cumulative gain",
        subplot: bool = False,
    ) -> None:
        metalearning_error = np.square(y_true - y_pred)
        baseline_error = np.square(y_true - y_baseline)
        mtl_gain = baseline_error - metalearning_error
        cumulative_gain = mtl_gain.cumsum()

        # plot
        if not subplot:
            _ = plt.figure(figsize=(25, 10))
        cumulative_gain.plot.area(stacked=False)

        print("Cumulative gain definition: squared_error(baseline) - squared_error(metalearning)")
        plt.xlabel("Meta learning batch")
        plt.ylabel("Cumulative gain")
        plt.title(title)

    def get_regression_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        r2 = evaluator.evaluate(y_true, y_pred, 'r2')
        mse = evaluator.evaluate(y_true, y_pred, 'mse')
        std = evaluator.evaluate(y_true, y_pred, 'std')
        return {'r2': r2, 'mse': mse, 'std': std}

evaluator = PerformanceEvaluator()
