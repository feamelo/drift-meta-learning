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

    def _get_scaled(self, df: pd.DataFrame) -> pd.DataFrame:
        # Filter numeric variables
        df = df.select_dtypes(include=np.number)
        scaler = StandardScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])
        return df

    def _get_dbscan_metrics (self, df: pd.DataFrame) -> dict:
        n = df.shape[0]
        db = DBSCAN(**self.model_params).fit(df)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_prop = list(labels).count(-1) / n
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

    def _train_kmeans(self, df: pd.DataFrame) -> KMeans:
        # A list holds the SSE values for each k
        sse = []
        models = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, **self.model_params)
            kmeans.fit(df)
            models.append(kmeans)
            sse.append(kmeans.inertia_)

        kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
        return models[kl.elbow - 1]

    def _get_kmeans_metrics(self, df: pd.DataFrame) -> dict:
        kmeans = self._train_kmeans(df)
        compacness = self._get_kmeans_compactness(kmeans, df)
        return {
            'compacness': compacness,
            'kmeans_n_iter': kmeans.n_iter_,
            'kmeans_n_clusters': kmeans.n_clusters,
            'kmeans_inertia': kmeans.inertia_,
        }

    def fit(self, *args):
        return self

    def evaluate(self, df: pd.DataFrame) -> dict:
        df = self._get_scaled(df)
        if self.clustering_type == 'kmeans':
            return self._get_kmeans_metrics(df)
        return self._get_dbscan_metrics(df)