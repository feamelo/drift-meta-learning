import pandas as pd
import numpy as np
from typing import Tuple


# Macros
N_QUANTILES = 10

class PsiCalculator():
    def __init__(self):
        """Calculates the PSI (Population Stability Index) for univariate drift analysis
        https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html

        The PSI is a measure of how much that distribution is similar to the reference data
        (data used in the model training)

        Interpretation:
            * PSI < 0.1: No significant changes in distribution
            * PSI < 0.2: Moderate changes
            * PSI >= 0.2: Significant changes

        The class consists of two main methods:
            - fit: Calculates the deciles of each variable of training dataset
            - evaluate: Retrieves reference sample data and calculates the PSI
        """
        pass

    def fit(self, df: pd.DataFrame):
        """Calculates the deciles for each attribute of the dataframe

        Args:
            df (pandas.Dataframe): Reference dataset
        """
        self.ref_data = {}

        for col in df.columns:
            self.ref_data[col] = self._get_bins(df[col])
            deciles = self._get_ref_deciles(df[col], col)
            self.ref_data[col]['deciles'] = deciles
        return self

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retrieves reference sample data and calculates the PSI

        Args:
            x (pandas.Dataframe) Monitoring dataset

        Returns:
            dict: Dictionary containing the PSI for each variable
        """
        psi_dict = {}

        for col in df.columns:
            psi_dict[f'psi_{col}'] = self._get_psi(df, col)
        return psi_dict

    def _get_ref_deciles(self, x: pd.Series, var_name: str) -> dict:
        """Calculates the deciles of the reference variable

        Args:
            x (pandas.Series): Reference variable

        Returns:
            dict: Dictionary containing the list of deciles for each attribute of the model,
            the type of the variable (categorical or numeric) and the splits limits (bins)
            of the deciles for the numeric variables.
        """
        data = self.ref_data[var_name]

        if(self._is_categorical(x)):
            deciles = self._get_categories(x)
        else:
            deciles, _ = self._get_num_quantiles(x, data['bins'])
            deciles = list(deciles)
        return deciles

    def _get_bins(self, x: pd.Series) -> dict:
        """Calculates the bins of the reference variable.
        - For numerical variables: deciles splits
        - For categorical variables: list of categories

        Args:
            x (pandas.Series): Reference variable

        Returns:
            Dictionary containing the type of the variable (categorical or numerical),
            the deciles splits for the numeric variables and the list of categories for
            the categorical variables
        """
        data = {}

        if(self._is_categorical(x)):
            data['var_type'] = 'cat'
            data['bins'] = list(x.unique())
        else:
            _, bins = self._get_num_quantiles(x)
            data['var_type'] = 'num'
            data['bins'] = list(bins)
        return data

    def _get_psi(self, df: pd.DataFrame, var_name: str) -> float:
        """Retrieves reference data from .fit method and calculates
        the PSI for the specified variable

        Args:
            df (pandas.Dataframe): Monitoring dataframe
            var_name (string): Monitoring variable name

        Returns:
            float: Population Stability Index of the variable with the specified var_name
        """
        x = df[var_name]
        reference = self.ref_data[var_name]

        if(reference['var_type'] == 'cat'):
            return self._get_cat_psi(x, reference)
        return self._get_num_psi(x, reference)

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

    def _get_cat_psi(self, x: pd.Series, reference: dict) -> float:
        """Calculate the PSI value of a categorical variable

        Args:
            x (pd.Series): Monitoring variable
            reference (dict): Reference data 

        Returns:
            float: Population Stability Index of the variable
        """
        monit = self._get_categories(x)
        ref = reference['deciles']
        return sum(self._sub_psi(monit[k], ref[k]) for k in monit.keys())

    def _get_num_psi(self, x: pd.Series, reference: dict) -> float:
        """Calculate the PSI value of a numerical variable

        Args:
            x (pd.Series) Monitoring variable
        reference : objeto
            reference (dict): Reference data 

        Returns:
            float: Population Stability Index of the variable
        """
        monit_deciles, _ = self._get_num_quantiles(x, reference['bins'])
        ref_deciles = reference['deciles']
        N = len(monit_deciles)
        return sum(self._sub_psi(monit_deciles[i], ref_deciles[i]) for i in range(N))

    def _get_splits(self, x: pd.Series, n_quantiles: Tuple[int, list, np.array]=N_QUANTILES) -> list:
        """Calculates the upper and lower bounds of quantiles.
        If there are equal bounds, reduce the amount of quantiles for the variable.

        Args:
            x (pd.Series): Variable to calculate splits
            n_quantiles (int): If integer, determines the number of quantiles, otherwise it is the splits list

        Returns:
            list: Upper and lower bounds of quantiles
        """
        if (type(n_quantiles) != int):
            return n_quantiles

        step = int(100/n_quantiles)
        bins = np.percentile(x, np.arange(step, 100 + n_quantiles, step))
        qtd_unique = pd.Series(bins).nunique()

        if(qtd_unique != len(bins)):
            return self._get_splits(x, qtd_unique)
        return bins

    def _get_num_quantiles(self, x: pd.Series, n_quantiles: int=N_QUANTILES) -> Tuple[list, list]:
        """Calculates the quantiles for a numerical variable

        Args:
            x (pd.Series): Variable to calculate the quantiles
            n_quantiles (int): Number of quantiles used for calculating PSI

        Returns:
            scores (list): The percentage value of each quantile for the variable x
            bins (list): Upper and lower bounds of quantiles
        """
        bins = self._get_splits(x, n_quantiles)
        scores, bins = np.histogram(x, bins=bins)
        scores = np.divide(scores, len(x))
        return scores, bins

    def _get_categories(self, x):
        """Calculates the percentage occurrence of each category

        Args:
            x (pd.Series): Categorical variable

        Returns:
            dict: Dictionary containing the categories (possible values
            of the variable x) as keys and the percentage occurrence of each
            category as value.
        """
        df = pd.DataFrame()
        df['categories'] = x.astype(str)
        df = df.value_counts(normalize=True).to_frame().reset_index()
        return df.set_index('categories').to_dict()[0]

    def _is_binary(self, x: pd.Series) -> bool:
        """Checks whether a given variable is binary or not.

        Args:
            x (pd.Series): variable

        Returns:
            Boolean indicating whether the variable x is binary or not
        """
        uniques = x.unique()
        if len(uniques) == 2 and (0 in uniques) and (1 in uniques):
            return True
        return False

    def _is_categorical(self, x: pd.Series) -> bool:
        """Checks whether a given variable is categorical or numeric.

        Args:
            x (pd.Series): variable

        Returns:
            Boolean indicating whether the variable x is categorical (True)
            or numeric (False)
        """
        if x.dtype == 'O':
            return True
        if x.dtype == 'bool':
            return True
        if(x.dtype == 'float64') or (x.dtype == 'int64'):
            return self._is_binary(x)
        return True
