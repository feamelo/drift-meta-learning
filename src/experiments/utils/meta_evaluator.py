import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
from meta_learning import evaluator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# Macros
COLORS = [
    "#eb5600ff", # orange
    "#1a9988ff", # green
    "#595959ff", # grey
    "#6aa4c8ff", # blue
    "#f1c232ff", # yellow
    ]
BASE_MODELS = [
    "RandomForestClassifier",
    "SVC",
    "LogisticRegression",
    "DecisionTreeClassifier"
]


class MetaEvaluator():
    def __init__(self, dataset_name: str, window_size: int = 30, feature_fraction: int=100):
        self.window_size = window_size
        self.dataset_name = dataset_name
        self.feature_fraction = feature_fraction

    def _get_mean_mse(self, cols: list, data_frame: pd.DataFrame, metric: str=None):
        result_mse = pd.DataFrame(columns=cols)
        if not metric:
            metric = cols[0].split("_")[0]

        iter_range = range(0, data_frame.shape[0] - self.window_size, self.window_size)
        for result_idx, original_idx in enumerate(iter_range):
            batch = data_frame.iloc[original_idx:original_idx + self.window_size]
            for col in cols:
                result_mse.loc[result_idx, col] = mean_squared_error(batch[metric], batch[col])
        return result_mse.mean(axis=1)

    def _get_result_df(self, filename: str):
        df = pd.read_csv(filename).dropna()
        metrics = list(set(df.columns).intersection(set(evaluator.binary_clf_metrics)))
        results = pd.DataFrame()
        for metric in metrics:
            metric_cols = [col for col in df.columns if metric in col]
            with_drift_cols = [col for col in metric_cols if "with_drift" in col]
            without_drift_cols = [col for col in metric_cols if "without_drift" in col]
            results[f"{metric}_proposed_mtl_mse"] = self._get_mean_mse(with_drift_cols, df)
            results[f"{metric}_original_mtl_mse"] = self._get_mean_mse(without_drift_cols, df)
            results[f"{metric}_mse_baseline"] = self._get_mean_mse([f"last_{metric}"], df, metric)
        return results, metrics

    def _plot_subplot(self, results_df: pd.DataFrame, color: str=COLORS[0], metric="kappa"):
        proposed_mtl_error = results_df[f"{metric}_proposed_mtl_mse"]
        baseline_error = results_df[f"{metric}_mse_baseline"]
        proposed_mtl_gain = baseline_error - proposed_mtl_error

        y = proposed_mtl_gain.cumsum()
        x = np.arange(len(y))
        plt.fill_between(x, y, alpha=0.1, color=color)
        plt.plot(x, y, label=metric, color=color)
        plt.legend(loc=2, fontsize='large')

    def fit(self):
        self.results = {}
        self.metrics = {}
        for base_model in BASE_MODELS:
            filename = f"results_dataframes/base_model: {base_model} - dataset: {self.dataset_name} - select_k_features: {self.feature_fraction}%.csv"
            self.results[base_model], self.metrics[base_model] = self._get_result_df(filename)
        return self

    def plot_gain(self):
        plt.figure(figsize=(25, 5))
        plt.suptitle(f"dataset: {self.dataset_name}")
        for base_model_idx, base_model in enumerate(BASE_MODELS):
            plt.subplot(1, 4, base_model_idx + 1)
            for metric_idx, metric in enumerate(self.metrics[base_model]):
                plt.title(base_model, fontsize=20)
                self._plot_subplot(self.results[base_model], metric=metric, color=COLORS[metric_idx])

    def _plot_comp_subplot(self, results_df: pd.DataFrame, color: str=COLORS[0], metric: str="kappa", plot_col: str="proposed_mtl"):
        if plot_col == "ideal_regressor":
            results_df[f"{metric}_ideal_regressor_mse"] = 0
        regressor_error = results_df[f"{metric}_{plot_col}_mse"]
        baseline_error = results_df[f"{metric}_mse_baseline"]
        proposed_mtl_gain = baseline_error - regressor_error

        y = proposed_mtl_gain.cumsum()
        x = np.arange(len(y))
        plt.fill_between(x, y, alpha=0.1, color=color)
        plt.plot(x, y, label=plot_col, color=color)
        plt.legend(loc=2, fontsize='large')

    def plot_original_vs_proposed_mtl_gain(self, metric="kappa", plot_ideal_regressor=True):
        plt.figure(figsize=(25, 5))
        plt.suptitle(f"dataset: {self.dataset_name} - metric: {metric}")
        for base_model_idx, base_model in enumerate(BASE_MODELS):
            plt.subplot(1, 4, base_model_idx + 1)
            if plot_ideal_regressor:
                self._plot_comp_subplot(self.results[base_model], metric=metric, color=COLORS[2], plot_col="ideal_regressor")
            self._plot_comp_subplot(self.results[base_model], metric=metric, color=COLORS[0], plot_col="proposed_mtl")
            self._plot_comp_subplot(self.results[base_model], metric=metric, color=COLORS[1], plot_col="original_mtl")
            plt.title(base_model, fontsize=20)
