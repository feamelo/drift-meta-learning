from typing import Tuple
import pandas as pd
from scipy import stats


# Macros
DEFAULT_DRIFT_THRESHOLD = 0.001
DRIFT = 1
NO_DRIFT = 0


class SqsiCalculator():
    """Implementation of 'Stream Quantification by Score Inspection' drift metric
    (Maletzke, dos Reis, & Batista, 2018).
    The KS (Kolmogorov Smirnov) test is applied for each score column (output of
    predict_proba method of base models, it can result in multiple columns in case of
    multiclass problems). If the null hypothesis is rejected, a linear transformation is
    applied to the reference data such that the instances in the window have the same mean
    and standard deviation as the detection data. After that the KS test is applied again,
    if the null hypothesis is rejected, then drift is detected.

    Args:
        score_cols (str): Name of dataframe columns containing the base model score
        drift_threshold (float, optional): Minimum TODO percentage of change in cluster distance
            to be considered as drift, if any cluster has a distance difference greater than
            this threshold, the drift flag will be True. Defaults to 0.001 as reccomended by
                the paper authors.
    """
    def __init__(
        self,
        score_cols: Tuple[list, str],
        drift_threshold: float=DEFAULT_DRIFT_THRESHOLD,
    ):
        if isinstance(score_cols, str):
            score_cols = [score_cols]
        self.score_cols = score_cols
        self.drift_threshold = drift_threshold
        self.ref_scores = {}

    def _check_drift(self, scores_ks: dict) -> int:
        """Check if any of the KS pvalues is smaller than the defined thresholds.
        If the null hypothesis is rejected, drift is detected.

        Args:
            scores_ks (dict): Dictionary containing the scores KS

        Returns:
            int: Flag indicating wether the data has drift or not
        """
        pvalues = [key for key in scores_ks.keys() if "pvalue" in key]
        for pvalue in pvalues:
            if scores_ks[pvalue] < self.drift_threshold:
                return DRIFT
        return NO_DRIFT

    def _linear_transformation(
        self,
        ref_score: pd.Series,
        monit_score: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Linear transformation to make the reference score have the same
        mean and standard deviation as the monitoring score. Formula:

        m1, s1: existing mean and std (reference mean/std)
        m2, s2: desired mean and std (monitoring mean/std)
        new = m2 + (ref - m1) * s2/s1

        Args:
            ref_score (pd.Series): Reference score
            monit_score (pd.Series): Monitoring score

        Returns:
            Tuple[pd.Series, pd.Series]: Reference and monitoring scores with
                the same mean and standard deviation
        """
        ref_mean, ref_std = ref_score.mean(), ref_score.std()
        monit_mean, monit_std = monit_score.mean(), monit_score.std()
        new_ref_score = monit_mean + (ref_score - ref_mean) * monit_std/ref_std
        return new_ref_score, monit_score

    def _get_ks(self, ref_score: pd.Series, monit_score: pd.Series) -> dict:
        statistic, pvalue = stats.kstest(ref_score, monit_score)
        if pvalue < self.drift_threshold:
            ref_score, monit_score = self._linear_transformation(ref_score, monit_score)
            statistic, pvalue = stats.kstest(ref_score, monit_score)
        return statistic, pvalue

    def _get_scores_ks(self, monit_scores: pd.DataFrame) -> dict:
        """Calculate KS for each scoring column of the dataframe comparing
        reference and monitoring distributions and save the result in a dictionary

        Args:
            monit_scores (pd.DataFrame): Scores of monitoring data

        Returns:
            dict: Dictionary containing KS statistic and pvalue for each scoring col
        """
        scores_ks = {}
        for col in self.ref_scores.columns:
            statistic, pvalue = self._get_ks(self.ref_scores[col], monit_scores[col])
            scores_ks[f"{col}_ks_statistic"] = statistic
            scores_ks[f"{col}_ks_pvalue"] = pvalue
        return scores_ks

    def fit(self, data_frame: pd.DataFrame):
        """Salve scoring cols of reference dataframe as as internal property
        for latter use

        Args:
            data_frame (pd.DataFrame): Reference dataframe
        """
        self.ref_scores = data_frame[self.score_cols]
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
        scores_ks = self._get_scores_ks(data_frame)
        drift_flag = self._check_drift(scores_ks)
        return {**scores_ks, "sqsi_drift_flag": drift_flag}
