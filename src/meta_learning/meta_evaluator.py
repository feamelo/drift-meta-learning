import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


# Macros
BASELINE = "baseline"
MTL = "meta_learning"
COLORS = ["#a2c4c9", "#facb9c"]


class MetaEvaluator():
    def __init__(
        self,
        baseline_col: str,
        mtl_col: str,
        target_col: str,
        eta: int = 50,
        step: int = 5
    ):
        self.eta = eta
        self.step = step
        self.baseline_col = baseline_col
        self.mtl_col = mtl_col
        self.target_col = target_col
        self.results_df = pd.DataFrame()

    def fit(self, meta_base: pd.DataFrame):
        df_size = meta_base.shape[0]
        results = []

        for t in range(0, df_size - self.eta, self.step):
            window_df = meta_base.iloc[t:t + self.eta]
            y_true = window_df[self.target_col]
            y_baseline = window_df[self.baseline_col]
            y_mtl = window_df[self.mtl_col]

            results.append({
                f"{BASELINE}_r2": r2_score(y_true, y_baseline),
                f"{BASELINE}_mse": mean_squared_error(y_true, y_baseline),
                f"{MTL}_r2": r2_score(y_true, y_mtl),
                f"{MTL}_mse": mean_squared_error(y_true, y_mtl),
            })
        self.results_df = pd.DataFrame(results)
        return self

    def plot_cumulative_gain(
        self,
        title: str = "Cumulative gain",
        subplot_tuple: tuple = None,
        plot_perfect_eval: bool = True,
    ) -> None:
        metalearning_error = self.results_df[f"{MTL}_mse"]
        baseline_error = self.results_df[f"{BASELINE}_mse"]
        mtl_gain = baseline_error - metalearning_error
        cumulative_gain = mtl_gain.cumsum()

        # plot
        if subplot_tuple:
            plt.subplot(*subplot_tuple)
        else:
            plt.figure(figsize=(25, 10))

        if plot_perfect_eval:
            df_plot = pd.DataFrame()
            df_plot["meta_learning"] = cumulative_gain
            df_plot["optimal_regressor"] = baseline_error.cumsum()
            df_plot.plot.area(stacked=False, color=COLORS, ax=plt.gca())
        else:
            cumulative_gain.plot.area(stacked=False, color=COLORS[0], ax=plt.gca())

        print("Cumulative gain definition: mse(baseline) - mse(metalearning)")
        plt.xlabel("Meta learning batch")
        plt.ylabel("Cumulative gain")
        plt.title(title, fontsize=20)
        plt.legend(loc=2, prop={'size': 20})

        if plot_perfect_eval:
            mtl_final_gain = list(cumulative_gain)[-1]
            ideal_final_gain = list(df_plot["optimal_regressor"])[-1]
            return mtl_final_gain/ideal_final_gain

    def barchart_with_std(self, col_suffix: str="r2", title: str=""):
        baseline_data = self.results_df[f"{BASELINE}_{col_suffix}"]
        mtl_data = self.results_df[f"{MTL}_{col_suffix}"]
        x=[BASELINE, MTL]
        y=[baseline_data.mean(), mtl_data.mean()]
        std=[baseline_data.std(), mtl_data.std()]
        plt.bar(x, y, color=COLORS)
        plt.ylabel(col_suffix)
        std_props = {
            "fmt": ".",
            "color": "Black",
            "elinewidth": 2,
            "capthick": 10,
            "errorevery": 1,
            "alpha": 0.5,
            "ms": 4,
            "capsize": 2
        }
        plt.errorbar(x, y, std, **std_props)
        plt.title(f"{col_suffix} {title}", fontsize=20)

    def plot_results(
        self,
        figsize: tuple=(25, 10),
        subplot_idx: int=1,
        subplot_shape: tuple=(1, 2),
        title: str="",
    ):
        if figsize:
            plt.figure(figsize=figsize)
        plt.rc('xtick',labelsize=20)
        plt.subplot(*subplot_shape, subplot_idx)
        self.barchart_with_std("mse", title)
        plt.subplot(*subplot_shape, subplot_idx + 1)
        self.barchart_with_std("r2", title)
