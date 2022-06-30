import numpy as np
import pandas as pd
from typing import Tuple

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit


# Macros
DEFAULT_N_FOLDS = 5
DEFAULT_MODEL = RandomForestClassifier
DEFAULT_METRIC = "precision"
DEFAULT_SCORING_STRATEGY = "max"
POSSIBLE_SCORING_STRATEGY = ["max", "min"]
DEFAULT_CROSS_VAL_TYPE = "time_series"
POSSIBLE_CROSS_VAL_TYPES = ["time_series", "kfold"]
VERBOSE = False
R_STATE = 2022


class Model():
    def __init__(
        self,
        basis_model = DEFAULT_MODEL,
        hyperparameters: Tuple[list, dict] = [],
        n_folds: int = DEFAULT_N_FOLDS,
        scoring_metric: str = DEFAULT_METRIC,
        scoring_strategy: str = DEFAULT_SCORING_STRATEGY,
        cross_val_type: str = DEFAULT_CROSS_VAL_TYPE,
        verbose: bool = VERBOSE,
        ):
        if scoring_metric not in sklearn.metrics.SCORERS.keys():
            raise Exception(f"Invalid scoring metric, must be one of: {sklearn.metrics.SCORERS.keys()}")
        if scoring_strategy not in POSSIBLE_SCORING_STRATEGY:
            raise Exception(f"Invalid scoring strategy, must be one of: {POSSIBLE_SCORING_STRATEGY}")
        if cross_val_type not in POSSIBLE_CROSS_VAL_TYPES:
            raise Exception(f"Invalid cross validation type, must be one of: {POSSIBLE_CROSS_VAL_TYPES}")

        self.__dict__.update(locals())

    def _kfold_cv(self, X: pd.DataFrame, y: pd.Series, model) -> float:
        return cross_val_score(model, X, y, cv=self.n_folds, scoring=self.scoring_metric)

    def _time_series_cv(self, X: pd.DataFrame, y: pd.Series, model) -> float:
        cv = TimeSeriesSplit(n_splits=self.n_folds).split(X)
        return cross_val_score(model, X, y, cv=cv, scoring=self.scoring_metric)

    def _cross_validation(self, X: pd.DataFrame, y: pd.Series) -> dict:
        if type(self.hyperparameters) == dict:
            return self.hyperparameters
        if not self.hyperparameters:
            return {}

        scores_list = []
        for idx, hyper in enumerate(self.hyperparameters):
            model = self.basis_model(**{**{"random_state": R_STATE}, **hyper})

            if self.cross_val_type == "kfold":
                scores = self._kfold_cv(X, y, model)
            else:
                scores = self._time_series_cv(X, y, model)
            scores_list.append(scores.mean())

            if self.verbose:
                print(f"Hyperparameters {idx}: {scores.mean()} {self.scoring_metric} with a standard deviation of {scores.std()}")

        best_idx = getattr(np, f'arg{self.scoring_strategy}')(scores_list)
        return self.hyperparameters[best_idx]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        best_hyperparams = self._cross_validation(X, y)
        self.best_hyperparams = {**{"random_state": R_STATE}, **best_hyperparams}
        self.model = self.basis_model(**self.best_hyperparams).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict_proba(X)

    def partial_fit(self, X: pd.DataFrame, y: pd.Series):
        """Only available for lightGBM model"""
        if not hasattr(self, 'model'):
            self.fit(X, y)
        else:
            init_model = self.model.booster_
            self.model = self.basis_model(**self.best_hyperparams).fit(X, y, init_model=init_model)
