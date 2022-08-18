import pandas as pd


# Macros
SEED = 2020
DEFAULT_DRIFT_THRESHOLD = 0.3
DRIFT = 1
NO_DRIFT = 0


class Udetector():
    """Implementation of 'UDetect: Unsupervised Concept Change Detection for Mobile
    Activity Recognition' paper proposed by Bashir, Sulaimon A. and Petrovski, Andrei
    and Doolan, Daniel.

    Args:
        prediction_col (str): Name of dataframe column containing the base model prediction
        drift_threshold (float, optional): Minimum percentage of change in cluster distance
            to be considered as drift, if any cluster has a distance difference greater than
            this threshold, the drift flag will be True. Defaults to 0.3
    """
    def __init__(self, prediction_col: str, drift_threshold: float=DEFAULT_DRIFT_THRESHOLD):
        self.prediction_col = prediction_col
        self.drift_threshold = drift_threshold
        self.thresholds = {}

    def _get_cluster_distance(self, data_frame: pd.DataFrame) -> float:
        """Calculate the distance to cluster centroid over all variables
        of the input dataframe.

        Args:
            data_frame (pd.DataFrame): Dataframe filtered with a single
                class instances
        Returns:
            float: Average distance to centroid
        """
        df_count = len(data_frame)
        dist = 0
        for feature in data_frame.columns:
            centroid = data_frame[feature].mean()
            dist += (data_frame[feature] - centroid).pow(2).sum()/df_count
        return dist/len(data_frame.columns)

    def _get_total_distance(self, data_frame: pd.DataFrame) -> dict:
        """Gets the average distance to centroid for each of the base model classes.

        Args:
            data_frame (pd.DataFrame): Input dataframe

        Returns:
            dict: Dictionary containing the distances (values) and classes (keys)
        """
        clusters = data_frame[self.prediction_col].unique()
        dist = {}
        for cluster in clusters:
            # Filter data with specific label
            cluster_df = data_frame[data_frame[self.prediction_col] == cluster]
            cluster_df.drop(self.prediction_col, axis=1, inplace=True)
            dist[f"distance_class_{cluster}"] = self._get_cluster_distance(
                cluster_df)
        return dist

    def _get_thresholds(self, distances: dict) -> dict:
        # TODO: apply heuristics for threshold calculation
        thresholds = {}
        for cluster, dist in distances.items():
            min_val = dist * self.drift_threshold
            max_val = dist * (1 + self.drift_threshold)
            thresholds[cluster] = {"min": min_val, "max": max_val}
        return thresholds

    def _check_drift(self, distances: dict) -> int:
        """Check if any of the clusters average distances is greater
        than the pre defined thresholds of the reference data.

        Args:
            distances (dict): Dictionary containing the monitoring data
                clusters distances

        Returns:
            int: Flag indicating wether the data has drift or not
        """
        # Get all possible clusters
        non_existing_thresh = {"min": 0, "max": 0}
        clusters = {**distances, **self.thresholds}.keys()
        for cluster in clusters:
            thresh = self.thresholds.get(cluster, non_existing_thresh)
            dist = distances.get(cluster, 0)
            if dist < thresh["min"] or dist > thresh["max"]:
                return DRIFT
        return NO_DRIFT

    def fit(self, data_frame: pd.DataFrame):
        """Fit reference dataframe by calculating the average distance
        to centroid for each base model class and creating a set of
        thresholds to find wether drift has occured or not.

        Args:
            data_frame (pd.DataFrame): Reference dataframe
        """
        distances = self._get_total_distance(data_frame)
        self.thresholds = self._get_thresholds(distances)
        return self

    def evaluate(self, data_frame: pd.DataFrame) -> dict:
        """Evaluate drift in monitoring dataframe.
        Calculates the average distance to centroid for each base model
        class and compares them to the pre defined threshold to find wether
        drift has occured or not.

        Args:
            data_frame (pd.DataFrame): Monitoring dataframe

        Returns:
            dict: Dictionary containing the distances for each class and
                the drift flag
        """
        distances = self._get_total_distance(data_frame)
        drift_flag = self._check_drift(distances)
        return {**distances, "u_detect_drift_flag": drift_flag}
