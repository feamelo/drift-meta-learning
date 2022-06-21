import pandas as pd
from typing import Tuple


# Macros
KNOWN_TARGET_DELAY = 3
KNOWN_TARGET_WINDOW_SIZE = 0
R_STATE = 2022
VERBOSE = False
DEFAULT_TYPE = "constant"
POSSIBLE_TYPES = ["constant"]
ETA = 100  # Window size used to extract meta features
STEP = 10  # Step for next meta learning iteration


class Baseline():
    def __init__(
        self,
        type: str = DEFAULT_TYPE,
        eta: int = ETA,
        step: int = STEP,
        known_target_delay: int = KNOWN_TARGET_DELAY,
        known_target_window_size: int = KNOWN_TARGET_WINDOW_SIZE,
        verbose: bool = VERBOSE,
    ):
        kwargs = locals()
        kwargs = {key: kwargs[key] for key in list(
            kwargs.keys()) if key not in ('self', '__class__')}
        self._update_params(**kwargs)

    def _update_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def fit(self, df: pd.DataFrame) -> None:
        """Fit offline data"""
        pass

    def update(self, df: pd.DataFrame) -> None:
        """Update baseline with new online data"""
        pass

    def target(self, y: Tuple[int, float, str]) -> None:
        """Update baseline with upcoming target"""
        pass
