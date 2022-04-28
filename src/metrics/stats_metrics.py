import pandas as pd
import numpy as np


# Macros
METRICS = ['precision', 'recall', 'f1-score', 'auc', 'kappa']

class StatsMetrics():
    def __init__(self):
        pass

    def _get_statistical_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        metrics = df.describe().reset_index()
        metrics = metrics[metrics['index'].isin(['mean', 'max', 'min', 'std'])]
        metrics = pd.melt(metrics, id_vars=['index'])
        metrics['variable'] = metrics['variable'] + '_' + metrics['index']
        return metrics.drop('index', axis=1)

    def _get_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        corr = df.corr()
        keep = np.logical_not(np.triu(np.ones(corr.shape))).astype('bool').reshape(corr.size)
        corr = pd.melt(corr.reset_index(), id_vars=['index'])[keep]
        corr['variable'] = 'correlation_' + corr['variable'] + '_' + corr['index']
        return corr.drop('index', axis=1)

    def fit(self, *args):
        return self

    def evaluate(self, df: pd.DataFrame) -> float:
        stats = self._get_statistical_metrics(df)
        corr = self._get_correlation(df)
        metrics = pd.concat([stats, corr], ignore_index=True).set_index('variable').T
        return metrics.reset_index(drop=True)