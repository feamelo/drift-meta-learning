import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score


# Macros
N_FOLDS = 5
SEED = 2020
REF_LABEL = 0
MONIT_LABEL = 1
R_STATE = 1
LABEL_COL = 'label'
LGBM_PARAMS = {'max_depth': 6, 'reg_alpha': 15, 'random_state': R_STATE}
DEFAULT_DRIFT_THRESHOLD = 0.3
DRIFT = 1
NO_DRIFT = 0

class DomainClassifier():
    """ Implementation of 'domain classifier', a multivariate drift metric proposed by Simona Maggio
    and Du Phan in the white paper 'A Primer on Data Drift & Drift Detection Techniques'.
    Link: https://pages.dataiku.com/data-drift-detection-techniques

    The algorithm consists of training a binary classifier that must differentiate samples
    with a reference label (data without drift) and monitoring (may have drift).

    If the data distribution is very similar, the classifier will have difficulties to differentiate
    between samples and will present an accuracy close to 0.5, as the drift increases we have an
    increase or decreased accuracy.

    Args:
        drift_threshold (float): Minimum accuracy difference (compared to 0.5) to be considered as
        drift, if the |accuracy - 0.5| is greater than this threshold, the drift flag will be True.
    """
    def __init__(self, drift_threshold: float=DEFAULT_DRIFT_THRESHOLD):
        self.drift_threshold = drift_threshold
        self.reference = pd.DataFrame()

    def _check_drift(self, accuracy: float) -> int:
        """Compare the obtained accuracy with a pre defined threshold

        Args:
            accuracy (float): Accuracy of domain classifier

        Returns:
            int: Flag indicating wether the data has drift or not
        """
        acc = abs(accuracy - 0.5)
        if acc > self.drift_threshold:
            return DRIFT
        return NO_DRIFT

    def _get_accuracy(self, features: pd.DataFrame, target: pd.Series) -> float:
        """Train a binary classifier to differentiate monitoring and reference
        dataframes and calculates the prediction accuracy.

        Args:
            features (pd.DataFrame): Dataframe containing the base model features
            target (pd.Series): Target series which is a set of flags indicating
                if the data corresponds to monitoring or reference dataframes

        Returns:
            float: accuracy of the classifier
        """
        clf = LGBMClassifier(**LGBM_PARAMS)
        scores = cross_val_score(clf, features, target, cv=5, scoring='accuracy')
        return np.mean(scores)

    def _create_dataset(self, monit_df: pd.DataFrame) -> pd.DataFrame:
        """Create a balanced dataset containing reference and monitoring dataframes,
        create a column to diferentiate the data type (monit/ref)

        Args:
            monit_df (pd.DataFrame): Monitoring dataframe

        Returns:
            pd.DataFrame: Dataframe to train the binary classifier
        """
        monit_df[LABEL_COL] = MONIT_LABEL
        ref_df = self.reference.copy()

        # Count monitoring and reference dataframe sizes
        ref_count = ref_df.shape[0]
        monit_count = monit_df.shape[0]
        df_count = min(ref_count, monit_count)

        # Balance datasets and concatenate both
        monit_df = monit_df.sample(df_count, random_state=R_STATE)
        ref_df = ref_df.sample(df_count, random_state=R_STATE)
        final_df = pd.concat([ref_df, monit_df])

        # split target and features into two variables
        features = final_df.drop(LABEL_COL, axis=1)
        target = final_df[LABEL_COL]
        return features, target

    def fit(self, data_frame: pd.DataFrame):
        """Salve reference dataframe as as internal property
        for latter use
        Args:
            data_frame (pd.DataFrame): Reference dataframe
        """
        data_frame[LABEL_COL] = REF_LABEL
        self.reference = data_frame.copy()
        return self

    def evaluate(self, data_frame: pd.DataFrame) -> dict:
        """Train an classifier to diferentiate the monitoring and
        reference dataframes, if the accuracy of the classififer is
        much different than 0.5, drift is detected.

        Args:
            data_frame (pd.DataFrame): Monitoring dataframe

        Returns:
            dict: Dictionary containing the domain classifier accuracy
                and the drift flag.
        """
        features, target = self._create_dataset(data_frame.copy())
        accuracy = self._get_accuracy(features, target)
        drift_flag = self._check_drift(accuracy)
        return {"dc_accuracy": accuracy, "dc_drift_flag": drift_flag}
