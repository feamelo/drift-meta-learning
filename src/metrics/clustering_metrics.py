from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pandas as pd
import numpy as np


# Macros
CLUSTERING_TYPE = 'kmeans'
DBSCAN_PARAMS = {
    'eps': 0.3,
    'min_samples': 10,
}
KMEANS_PARAMS = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}


class ClusteringMetrics():
    def __init__(self, clustering_type: str='kmeans', model_params: dict=KMEANS_PARAMS):
        self.clustering_type = clustering_type
        self.model_params = model_params

    def _get_scaled(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        # Filter numeric variables
        data_frame = data_frame.select_dtypes(include=np.number)
        scaler = StandardScaler()
        data_frame[data_frame.columns] = scaler.fit_transform(data_frame[data_frame.columns])
        return data_frame

    def _get_dbscan_metrics (self, data_frame: pd.DataFrame) -> dict:
        df_size = data_frame.shape[0]
        dbscan = DBSCAN(**self.model_params).fit(data_frame)
        core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True
        labels = dbscan.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_prop = list(labels).count(-1) / df_size
        return {'dbscan_n_clusters': n_clusters, 'dbscan_noise_proportion': noise_prop}

    def _get_kmeans_compactness(self, kmeans: KMeans, df: pd.DataFrame) -> float:
        n_clusters = kmeans.n_clusters
        labels = kmeans.labels_
        compactness = 0

        for i in range(n_clusters):
            centroid_coord = kmeans.cluster_centers_[i]
            distances = np.square(df[labels == i] - centroid_coord)
            distance_sum = np.sum(np.sqrt(distances.sum(axis=1)))
            compactness += distance_sum
        return compactness

    def _train_kmeans(self, data_frame: pd.DataFrame) -> KMeans:
        # A list holds the SSE values for each k
        sse = []
        models = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, **self.model_params)
            kmeans.fit(data_frame)
            models.append(kmeans)
            sse.append(kmeans.inertia_)

        knee_locl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
        elbow = knee_locl.elbow or 1
        return models[elbow - 1]

    def _get_kmeans_metrics(self, data_frame: pd.DataFrame) -> dict:
        kmeans = self._train_kmeans(data_frame)
        compacness = self._get_kmeans_compactness(kmeans, data_frame)
        return {
            'compacness': compacness,
            'kmeans_n_iter': kmeans.n_iter_,
            'kmeans_n_clusters': kmeans.n_clusters,
            'kmeans_inertia': kmeans.inertia_,
        }

    # pylint: disable=unused-argument
    def fit(self, *args):
        return self

    def evaluate(self, df: pd.DataFrame) -> dict:
        df = self._get_scaled(df)
        if self.clustering_type == 'kmeans':
            return self._get_kmeans_metrics(df)
        return self._get_dbscan_metrics(df)
