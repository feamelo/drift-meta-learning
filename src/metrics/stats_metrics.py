import pandas as pd
import numpy as np
from scipy.stats import hmean, gmean, entropy
from sklearn.decomposition import PCA


# Macros
METRICS = ['precision', 'recall', 'f1-score', 'auc', 'kappa']
PCA_THRESH = 0.95

class StatsMetrics():
    def __init__(self):
        pass

    def _get_prop_pca(self, df: pd.DataFrame) -> dict:
        n = df.shape[0]
        pca = PCA(n_components=PCA_THRESH)
        principal_components = pca.fit_transform(df.select_dtypes(include=np.number))
        return {'prop_pca': principal_components.shape[1] / n}

    def _get_sparsity(self, df: pd.DataFrame) -> dict:
        n = df.shape[0]
        uniqueness_ratio = (df.nunique()/n).to_dict()
        sparsity = (1 - df.fillna(0).astype(bool).sum(axis=0) / n).to_dict()
        uniqueness_ratio = {k + '_uniqueness_ratio': v for k, v in uniqueness_ratio.items()}
        sparsity = {k + '_sparsity': v for k, v in sparsity.items()}
        return {**uniqueness_ratio, **sparsity}

    def _get_nr_outliers_iqr(self, df: pd.DataFrame) -> dict:
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1

        nr_out = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum().to_dict()
        iqr_dict = IQR.to_dict()
        nr_out = {k + '_nr_outliers': v for k, v in nr_out.items()}
        iqr_dict = {k + '_iqr': v for k, v in iqr_dict.items()}
        return {**nr_out, **iqr_dict}

    def _get_kurt_skew(self, df: pd.DataFrame) -> dict:
        skew = df.skew(axis=0).to_dict()
        kurt = df.kurt(axis=0).to_dict()
        return {**skew, **kurt}

    def _get_gmean_hmean_entropy(self, df: pd.DataFrame) -> dict:
        metrics = {}
        for col in df.columns:
            metrics[f'{col}_gmean'] = gmean(df[col])
            metrics[f'{col}_hmean'] = hmean(df[col])
            metrics[f'{col}_entropy'] = entropy(df[col])
            metrics[f'{col}_median'] = df[col].median()
        return metrics

    def _get_statistical_metrics(self, df: pd.DataFrame) -> dict:
        metrics = df.describe().reset_index()
        metrics = metrics[metrics['index'].isin(['mean', 'max', 'min', 'std'])]
        metrics = pd.melt(metrics, id_vars=['index'])
        metrics['variable'] = metrics['variable'] + '_' + metrics['index']
        metrics = metrics.drop('index', axis=1).set_index('variable').T
        return metrics.to_dict(orient='records')[0]

    def _get_correlation(self, df: pd.DataFrame) -> dict:
        corr = df.corr()
        keep = np.logical_not(np.triu(np.ones(corr.shape))).astype('bool').reshape(corr.size)
        corr = pd.melt(corr.reset_index(), id_vars=['index'])[keep]
        corr['variable'] = 'correlation_' + corr['variable'] + '_' + corr['index']
        corr = corr.drop('index', axis=1).set_index('variable').T
        return corr.to_dict(orient='records')[0]

    def fit(self, *args):
        return self

    def evaluate(self, df: pd.DataFrame) -> dict:
        # Filter numeric variables
        df = df.select_dtypes(include=np.number)

        stats = self._get_statistical_metrics(df)
        corr = self._get_correlation(df)
        means = self._get_gmean_hmean_entropy(df)
        outliers = self._get_nr_outliers_iqr(df)
        sparsity = self._get_sparsity(df)
        pca = self._get_prop_pca(df)
        return {**stats, **corr, **means, **outliers, **sparsity, **pca}