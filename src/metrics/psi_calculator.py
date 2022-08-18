import pandas as pd
import numpy as np
from typing import Tuple


# Macros
N_QUANTILES = 10
DEFAULT_DRIFT_THRESHOLD = 0.2
DRIFT = 1
NO_DRIFT = 0


class PsiCalculator():
    """Calculates the PSI (Population Stability Index) for univariate drift analysis
    The PSI is a measure of how much that distribution is similar to the reference data
    (data used in the model training)

    Interpretation:
        * PSI < 0.1: No significant changes in distribution
        * PSI < 0.2: Moderate changes
        * PSI >= 0.2: Significant changes

    The class consists of two main methods:
        - fit: Calculates the deciles of each variable of training dataset
        - evaluate: Retrieves reference sample data and calculates the PSI

    Args:
        input_cols (list): List of columns to calculate PSI. If not provided, all columns
            of reference dataframe will be used during .fit() call
        drift_threshold (float): Minimum PSI value to be considered as drift, if any variable
            has a psi greater than this threshold, the drift flag will be True.
    """
    def __init__(self, input_cols: list=None, drift_threshold: float=DEFAULT_DRIFT_THRESHOLD):
        self.ref_data = {}
        self.input_cols = input_cols
        self.drift_threshold = drift_threshold

    def fit(self, df: pd.DataFrame):
        """Calculates the deciles for each attribute of the dataframe

        Args:
            df (pandas.Dataframe): Reference dataset
        """
        if not self.input_cols:
            self.input_cols = df.columns
        for col in self.input_cols:
            self.ref_data[col] = self._get_bins(df[col])
            deciles = self._get_ref_deciles(df[col], col)
            self.ref_data[col]['deciles'] = deciles
        return self

    def evaluate(self, monit_df: pd.DataFrame) -> pd.DataFrame:
        """Retrieves reference sample data and calculates the PSI

        Args:
            x (pandas.Dataframe) Monitoring dataset

        Returns:
            dict: Dictionary containing the PSI for each variable
        """
        psi_dict = {}

        for col in monit_df.columns:
            psi_dict[f'psi_{col}'] = self._get_psi(monit_df, col)
        drift_flag = self._check_drift(psi_dict)
        return {**psi_dict, "psi_drift_flag": drift_flag}

    def _check_drift(self, psi_dict: dict) -> int:
        """Check if any variable has psi value higher than threshold

        Args:
            psi_dict (dict): Dictionary containing the psi (values)
                for each variable (keys)

        Returns:
            int: Flag indicating wether the drift has occured or not
        """
        if any(psi > self.drift_threshold for psi in psi_dict.values()):
            return DRIFT
        return NO_DRIFT

    def _get_ref_deciles(self, ref_feature: pd.Series, var_name: str) -> dict:
        """Calculates the deciles of the reference variable

        Args:
            ref_feature (pandas.Series): Reference variable

        Returns:
            dict: Dictionary containing the list of deciles for each attribute of the model,
            the type of the variable (categorical or numeric) and the splits limits (bins)
            of the deciles for the numeric variables.
        """
        data = self.ref_data[var_name]

        if self._is_categorical(ref_feature):
            deciles = self._get_categories(ref_feature)
        else:
            deciles, _ = self._get_num_quantiles(ref_feature, data["bins"])
            deciles = list(deciles)
        return deciles

    def _get_bins(self, ref_feature: pd.Series) -> dict:
        """Calculates the bins of the reference variable.
        - For numerical variables: deciles splits
        - For categorical variables: list of categories

        Args:
            ref_feature (pandas.Series): Reference variable

        Returns:
            Dictionary containing the type of the variable (categorical or numerical),
            the deciles splits for the numeric variables and the list of categories for
            the categorical variables
        """
        data = {}

        if self._is_categorical(ref_feature):
            data["var_type"] = "cat"
            data["bins"] = list(ref_feature.unique())
        else:
            _, bins = self._get_num_quantiles(ref_feature)
            data["var_type"] = "num"
            data["bins"] = list(bins)
        return data

    def _get_psi(self, monit_df: pd.DataFrame, var_name: str) -> float:
        """Retrieves reference data from .fit method and calculates
        the PSI for the specified variable

        Args:
            monit_df (pandas.Dataframe): Monitoring dataframe
            var_name (string): Monitoring variable name

        Returns:
            float: Population Stability Index of the variable with the specified var_name
        """
        feature = monit_df[var_name]
        reference = self.ref_data[var_name]

        if reference["var_type"] == "cat":
            return self._get_cat_psi(feature, reference)
        return self._get_num_psi(feature, reference)

    def _sub_psi(self, monit, ref):
        """Calculate the PSI value of a single quantile.
        Updates the actual value to a very small number if equal to zero

        Args:
            monit (float): Monit data decile
            ref (float): Reference data decile

        Returns:
            float: Decile PSI
        """
        if monit == 0:
            monit = 0.0001
        if ref == 0:
            ref = 0.0001
        return (monit - ref) * np.log(monit/ref)

    def _get_cat_psi(self, monit_feature: pd.Series, reference: dict) -> float:
        """Calculate the PSI value of a categorical variable

        Args:
            monit_feature (pd.Series): Monitoring variable
            reference (dict): Reference data

        Returns:
            float: Population Stability Index of the variable
        """
        monit = self._get_categories(monit_feature)
        ref = reference['deciles']
        return sum(self._sub_psi(monit[k], ref.get(k, 0)) for k in monit.keys())

    def _get_num_psi(self, monit_feature: pd.Series, reference: dict) -> float:
        """Calculate the PSI value of a numerical variable

        Args:
            monit_feature (pd.Series) Monitoring variable
        reference : objeto
            reference (dict): Reference data 

        Returns:
            float: Population Stability Index of the variable
        """
        monit_deciles, _ = self._get_num_quantiles(monit_feature, reference['bins'])
        ref_deciles = reference['deciles']
        n_deciles = len(monit_deciles)
        return sum(self._sub_psi(monit_deciles[i], ref_deciles[i]) for i in range(n_deciles))

    def _get_splits(
        self,
        feature: pd.Series,
        n_quantiles: Tuple[int, list, np.array]=N_QUANTILES
    ) -> list:
        """Calculates the upper and lower bounds of quantiles.
        If there are equal bounds, reduce the amount of quantiles for the variable.

        Args:
            feature (pd.Series): Variable to calculate splits
            n_quantiles (int): If integer, determines the number of quantiles,
                otherwise it is the splits list

        Returns:
            list: Upper and lower bounds of quantiles
        """
        if not isinstance(n_quantiles, int):
            return n_quantiles

        step = int(100/n_quantiles)
        bins = np.percentile(feature, np.arange(step, 100 + n_quantiles, step))
        qtd_unique = pd.Series(bins).nunique()

        if qtd_unique != len(bins):
            return self._get_splits(feature, qtd_unique)
        return bins

    def _get_num_quantiles(
        self,
        feature: pd.Series,
        n_quantiles: int=N_QUANTILES
    ) -> Tuple[list, list]:
        """Calculates the quantiles for a numerical variable

        Args:
            feature (pd.Series): Variable to calculate the quantiles
            n_quantiles (int): Number of quantiles used for calculating PSI

        Returns:
            scores (list): The percentage value of each quantile for the variable feature
            bins (list): Upper and lower bounds of quantiles
        """
        bins = self._get_splits(feature, n_quantiles)
        scores, bins = np.histogram(feature, bins=bins)
        scores = np.divide(scores, len(feature))
        return scores, bins

    def _get_categories(self, feature: pd.Series):
        """Calculates the percentage occurrence of each category

        Args:
            feature (pd.Series): Categorical variable

        Returns:
            dict: Dictionary containing the categories (possible values
            of the variable feature) as keys and the percentage occurrence of each
            category as value.
        """
        data_frame = pd.DataFrame()
        data_frame['categories'] = feature.astype(str)
        data_frame = data_frame.value_counts(normalize=True).to_frame().reset_index()
        return data_frame.set_index('categories').to_dict()[0]

    def _is_binary(self, feature: pd.Series) -> bool:
        """Checks whether a given variable is binary or not.

        Args:
            feature (pd.Series): variable

        Returns:
            Boolean indicating whether the variable feature is binary or not
        """
        uniques = feature.unique()
        if len(uniques) == 2 and (0 in uniques) and (1 in uniques):
            return True
        return False

    def _is_categorical(self, feature: pd.Series) -> bool:
        """Checks whether a given variable is categorical or numeric.

        Args:
            feature (pd.Series): variable

        Returns:
            Boolean indicating whether the variable feature is categorical (True)
            or numeric (False)
        """
        if feature.dtype == 'O':
            return True
        if feature.dtype == 'bool':
            return True
        if(feature.dtype == 'float64') or (feature.dtype == 'int64'):
            return self._is_binary(feature)
        return True
