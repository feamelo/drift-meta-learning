import pandas as pd
from typing import Tuple


# Macros
R_STATE = 2022
VERBOSE = False


class BaseLevelBase():
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
        self.new_row_id = 0
        self.new_target_id = 0
        self.new_batch_counter = 0
        self.new_target_batch_counter = 0
        self.base = pd.DataFrame()

    def get_batch(self) -> pd.DataFrame:
        self.new_batch_counter = 0
        batch = self.base.copy().drop(self.target_col, axis=1)
        return batch.tail(self.batch_size)

    def get_target_batch(self) -> pd.DataFrame:
        self.new_target_batch_counter = 0

        # Get a batch with last instances with known target
        batch = self.base.copy().dropna(subset=[self.target_col])
        return batch.tail(self.batch_size)

    def get_raw(self) -> pd.DataFrame:
        return self.base.copy()

    def fit(self, data_frame: pd.DataFrame) -> None:
        self.base = data_frame.copy().reset_index(drop=True)
        self.new_row_id = data_frame.shape[0]
        self.new_target_id = data_frame.shape[0]

    def update(self, new_line: pd.DataFrame) -> None:
        """Update base with new online data"""
        new_line["id"] = self.new_row_id
        new_line = new_line.set_index("id")
        self.base = pd.concat([self.base, new_line])
        self.new_row_id += 1
        self.new_batch_counter += 1

    def update_target(self, target: Tuple[int, float, str]) -> None:
        """Update base with upcoming target"""
        self.base.at[self.new_target_id, self.target_col] = target
        self.new_target_id += 1
        self.new_target_batch_counter += 1
