import pandas as pd
import numpy as np
from scipy.stats import hmean, gmean, entropy
from sklearn.decomposition import PCA


# Macros
PCA_THRESH = 0.95


class StatsMetrics():
    def __init__(self):
        pass

    def _get_prop_pca(self, data_frame: pd.DataFrame) -> dict:
        n = data_frame.shape[0]
        pca = PCA(n_components=PCA_THRESH)
        principal_components = pca.fit_transform(data_frame.select_dtypes(include=np.number))
        return {'prop_pca': principal_components.shape[1] / n}

    def _get_sparsity(self, data_frame: pd.DataFrame) -> dict:
        df_size = data_frame.shape[0]
        uniqueness_ratio = (data_frame.nunique()/df_size).to_dict()
        sparsity = (1 - data_frame.fillna(0).astype(bool).sum(axis=0) / df_size).to_dict()
        uniqueness_ratio = {k + '_uniqueness_ratio': v for k, v in uniqueness_ratio.items()}
        sparsity = {k + '_sparsity': v for k, v in sparsity.items()}
        return {**uniqueness_ratio, **sparsity}

    def _get_nr_outliers_iqr(self, data_frame: pd.DataFrame) -> dict:
        # pylint: disable=invalid-name
        Q1 = data_frame.quantile(0.25)
        Q3 = data_frame.quantile(0.75)
        IQR = Q3 - Q1

        nr_out = ((data_frame < (Q1 - 1.5 * IQR)) | (data_frame > (Q3 + 1.5 * IQR))).sum().to_dict()
        iqr_dict = IQR.to_dict()
        nr_out = {k + '_nr_outliers': v for k, v in nr_out.items()}
        iqr_dict = {k + '_iqr': v for k, v in iqr_dict.items()}
        return {**nr_out, **iqr_dict}

    def _get_kurt_skew(self, data_frame: pd.DataFrame) -> dict:
        skew = data_frame.skew(axis=0).to_dict()
        kurt = data_frame.kurt(axis=0).to_dict()
        return {**skew, **kurt}

    def _get_gmean_hmean_entropy(self, data_frame: pd.DataFrame) -> dict:
        metrics = {}
        for col in data_frame.columns:
            metrics[f'{col}_gmean'] = gmean(data_frame[col])
            metrics[f'{col}_hmean'] = hmean(data_frame[col])
            metrics[f'{col}_entropy'] = entropy(data_frame[col])
            metrics[f'{col}_median'] = data_frame[col].median()
        return metrics

    def _get_statistical_metrics(self, data_frame: pd.DataFrame) -> dict:
        metrics = data_frame.describe().reset_index()
        metrics = metrics[metrics['index'].isin(['mean', 'max', 'min', 'std'])]
        metrics = pd.melt(metrics, id_vars=['index'])
        metrics['variable'] = metrics['variable'] + '_' + metrics['index']
        metrics = metrics.drop('index', axis=1).set_index('variable').T
        return metrics.to_dict(orient='records')[0]

    def _get_correlation(self, data_frame: pd.DataFrame) -> dict:
        corr = data_frame.corr()
        keep = np.logical_not(np.triu(np.ones(corr.shape))).astype('bool').reshape(corr.size)
        corr = pd.melt(corr.reset_index(), id_vars=['index'])[keep]
        corr['variable'] = 'correlation_' + corr['variable'] + '_' + corr['index']
        corr = corr.drop('index', axis=1).set_index('variable').T
        return corr.to_dict(orient='records')[0]

    # pylint: disable=unused-argument
    def fit(self, *args):
        return self

    def evaluate(self, data_frame: pd.DataFrame) -> dict:
        # Filter numeric variables
        data_frame = data_frame.select_dtypes(include=np.number)

        stats = self._get_statistical_metrics(data_frame)
        corr = self._get_correlation(data_frame)
        means = self._get_gmean_hmean_entropy(data_frame)
        outliers = self._get_nr_outliers_iqr(data_frame)
        sparsity = self._get_sparsity(data_frame)
        pca = self._get_prop_pca(data_frame)
        return {**stats, **corr, **means, **outliers, **sparsity, **pca}
