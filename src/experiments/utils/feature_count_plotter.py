import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from meta_learning import evaluator


COLORS = [
    "#eb5600ff", # orange
    "#1a9988ff", # green
    "#595959ff", # grey
    "#6aa4c8ff", # blue
    "#f1c232ff", # yellow
    ]


class FeatureCountPlotter():
    def __init__(self, result_df: pd.DataFrame, group_col: str="feature_perc", dataset_col="dataset_name", base_model_col="base_model"):
        self.group_col = group_col
        self.dataset_col = dataset_col
        self.base_model_col = base_model_col
        self.result_df = result_df
        self.metrics = list(set(result_df.columns).intersection(set(evaluator.binary_clf_metrics)))

    def fit(self):
        group_values = self.result_df[self.group_col].unique()
        datasets = self.result_df[self.dataset_col].unique()
        base_models = self.result_df[self.base_model_col].unique()

        results = pd.DataFrame(columns=[
            "metric",
            self.group_col,
            self.base_model_col,
            self.dataset_col,
            "baseline_mse",
            "proposed_mtl_mse",
            "original_mtl_mse",
        ])

        for group in group_values:
            for dataset in datasets:
                for base_model in base_models:
                    df = self.result_df[
                        (self.result_df[self.group_col] == group) &
                        (self.result_df[self.dataset_col] == dataset) &
                        (self.result_df[self.base_model_col] == base_model)]

                    if df.shape[0]:
                        for metric in self.metrics:
                            results.loc[len(results)] = {
                                "metric": metric,
                                self.group_col: group,
                                self.dataset_col: dataset,
                                self.base_model_col: base_model,
                                "baseline_mse": mean_squared_error(df[metric], df[f"last_{metric}"]),
                                "proposed_mtl_mse": mean_squared_error(df[metric], df[f"{metric}_pred_0_with_drift"]),
                                "original_mtl_mse": mean_squared_error(df[metric], df[f"{metric}_pred_0_without_drift"]),
                            }
        self.summarized_df = results.set_index(self.group_col)
        return self

    def plot(self, title: str="", multiple_lines_col="metric", filter_col="dataset_name", ref_line_dict: dict=None, plot_legend=True, normalize=False):
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(title, fontsize=25)
        cols = ["proposed_mtl_mse", multiple_lines_col]

        for idx, filter_val in enumerate(self.summarized_df[filter_col].unique()):
            df_plot = self.summarized_df[self.summarized_df[filter_col] == filter_val][cols]
            df_plot = df_plot.groupby([self.group_col, multiple_lines_col]).sum().reset_index()
            df_plot = df_plot.pivot(values="proposed_mtl_mse", columns=multiple_lines_col, index=self.group_col)
            ax = fig.add_subplot(int(f"22{idx+1}"))

            if normalize:
                df_plot = (df_plot-df_plot.min())/(df_plot.max()-df_plot.min())

            if plot_legend:
                df_plot.plot(color=COLORS, lw=2, ax=ax, legend=True).legend(loc='upper right', prop={'size': 16})
            else:
                df_plot.plot(color=COLORS, lw=2, ax=ax, legend=False)
                
            if ref_line_dict:
                ax.axvline(ref_line_dict[filter_val], color='r', linestyle='--')
            plt.title(filter_val, fontsize=20)
