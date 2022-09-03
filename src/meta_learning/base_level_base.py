import pandas as pd
from typing import Tuple


# Macros
R_STATE = 2022
VERBOSE = False


class BaseLevelBase():
    """Class to manage the base level dataset.

    Args:
        batch_size (int):
            Meta learning window size
        target_col (str):
            Column name of target data
        prediction_col (str):
            Prediction column name
        verbose (bool, optional):
            Verbosity. Defaults to False.
    """
    def __init__(
        self,
        batch_size: int,
        target_col: str,
        prediction_col: str,
        verbose: bool = VERBOSE,
    ):
        # Update object properties with init params
        self.batch_size = batch_size
        self.target_col = target_col
        self.prediction_col = prediction_col
        self.verbose = verbose

        # Other properties
        self.new_target_id = 0
        self.new_batch_counter = 0
        self.new_target_batch_counter = 0
        self.base = pd.DataFrame()

    def get_batch(self) -> pd.DataFrame:
        """Get the last batch data"""
        self.new_batch_counter = 0
        batch = self.base.copy().drop(self.target_col, axis=1)
        return batch.tail(self.batch_size)

    def get_target_batch(self) -> pd.DataFrame:
        """Get batch of data with known target"""
        self.new_target_batch_counter = 0

        # Get a batch with last instances with known target
        batch = self.base.copy().dropna(subset=[self.target_col])
        return batch.tail(self.batch_size)

    def get_raw(self) -> pd.DataFrame:
        """Get the entire database"""
        return self.base.copy()

    def fit(self, data_frame: pd.DataFrame) -> None:
        """Save the offline database internally"""
        self.base = data_frame.copy().reset_index(drop=True)
        self.new_target_id = data_frame.shape[0]

    def update(self, new_line: pd.DataFrame) -> None:
        """Update base with new online data"""
        df_size = len(self.base)
        self.base.loc[df_size] = new_line
        self.new_batch_counter += 1

    def update_target(self, target: Tuple[int, float, str]) -> None:
        """Update base with upcoming target"""
        self.base.at[self.new_target_id, self.target_col] = target
        self.new_target_id += 1
        self.new_target_batch_counter += 1
