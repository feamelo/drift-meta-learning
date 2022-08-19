import pandas as pd
from itertools import combinations
from statistics import NormalDist
from river import drift


# Macros
DRIFT = 1
NO_DRIFT = 0


class OmvPht():
    """Implementation of the 'Online Modified Version of the Page-Hinkley Test',
    (Lughofer et al., 2016). It assumes that the uncertainty estimates of the
    two most probable classes corresponds to two gaussian distributions and the
    overlap of these two distributions is used as drift metric.

    Args:
        score_cols (str): Name of dataframe columns containing the base model scores
            null hypothesis, if the pvalue is smaller than this, drift is detected.
        page_hinkley_params (dict): Init params to be used in page hinkley test,
            check river documentation for params details:
            https://riverml.xyz/dev/api/drift/PageHinkley/
    """
    def __init__(self, score_cols: list, page_hinkley_params: dict={}):
        self.score_cols = score_cols
        self.score_cols_combination = list(combinations(score_cols, 2))
        self.page_hinkleys = {f"overlap_{col1}_{col2}": drift.PageHinkley(**page_hinkley_params)
                              for col1, col2 in self.score_cols_combination}

    def _check_drift(self, overlaps: dict) -> int:
        """Apply Page Hinkley test to scores overlap to check drift occurrence

        Args:
            overlaps (dict): Dictionary containing the scores overlaps

        Returns:
            int: Flag indicating wether the data has drift or not
        """
        for key, value in overlaps.items():
            self.page_hinkleys[key].update(value)
            if self.page_hinkleys[key].change_detected:
                return DRIFT
        return NO_DRIFT

    def _get_overlap(self, score_1: pd.Series, score_2: pd.Series) -> float:
        """Assume that the score distributions can be modeled as two gaussian
        distributions, get mean and standard deviation for each of them and
        calculathe the overlap between the two gaussian curves.

        Args:
            score_1 (pd.Series): Score distribution of first class
            score_2 (pd.Series): Score distribution of second class

        Returns:
            float: Overlap between the two distributions
        """
        mu1, std1 = score_1.mean(), score_1.std()
        mu2, std2 = score_2.mean(), score_2.std()
        norm1 = NormalDist(mu=mu1, sigma=std1)
        norm2 = NormalDist(mu=mu2, sigma=std2)
        if std1 == 0 or std2 == 0:
            return 0
        return norm1.overlap(norm2)

    def _get_scores_overlaps(self, monit_scores: pd.DataFrame) -> dict:
        """TODO doc Calculate overlaps for each scoring column pair

        Args:
            monit_scores (pd.DataFrame): Scores of monitoring data

        Returns:
            dict: Dictionary containing KS statistic and pvalue for each scoring col
        """
        score_overlaps = {}
        for col1, col2 in self.score_cols_combination:
            overlap = self._get_overlap(monit_scores[col1], monit_scores[col2])
            score_overlaps[f"overlap_{col1}_{col2}"] = overlap
        return score_overlaps

    def fit(self, data_frame: pd.DataFrame):
        """Fit reference data by initializing the Page Hinkley algorithm
        with initial score overlaps

        Args:
            data_frame (pd.DataFrame): Reference dataframe
        """
        overlaps = self._get_scores_overlaps(data_frame)
        self._check_drift(overlaps)
        return self

    def evaluate(self, data_frame: pd.DataFrame) -> dict:
        """Evaluate drift in monitoring dataframe.
        Apply Page Hinkley test to scores overlap to check drift occurrence

        Args:
            data_frame (pd.DataFrame): Monitoring dataframe

        Returns:
            dict: Dictionary containing the scoring overlaps and
                the drift flag
        """
        overlaps = self._get_scores_overlaps(data_frame)
        drift_flag = self._check_drift(overlaps)
        return {**overlaps, "omv_pth_drift_flag": drift_flag}
