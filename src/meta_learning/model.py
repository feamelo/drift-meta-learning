import numpy as np
import pandas as pd
from typing import Tuple

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing


# Macros
DEFAULT_N_FOLDS = 5
DEFAULT_MODEL = RandomForestClassifier
DEFAULT_METRIC = 'precision'
DEFAULT_SCORING_STRATEGY = 'max'
VERBOSE = False
R_STATE = 2022


class Model():
    def __init__(
        self,
        basis_model=DEFAULT_MODEL,
        hyperparameters: Tuple[list, dict]=[],
        n_folds: int=DEFAULT_N_FOLDS,
        scoring_metric: str=DEFAULT_METRIC,
        scoring_strategy: str=DEFAULT_SCORING_STRATEGY,
        verbose: bool=VERBOSE,
        ):
        if scoring_metric not in sklearn.metrics.SCORERS.keys():
            raise Exception(f"Invalid scoring metric, must be one of: {sklearn.metrics.SCORERS.keys()}")
        if scoring_strategy not in ['max', 'min']:
            raise Exception(f"Invalid scoring strategy, must be one of: {['max', 'min']}")

        self.__dict__.update(locals())

    def _cross_validation(self, X: pd.DataFrame, y: pd.Series) -> dict:
        if type(self.hyperparameters) == dict:
            return self.hyperparameters
        if not self.hyperparameters:
            return {}

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        scores_list = []
        for idx, hyper in enumerate(self.hyperparameters):
            model = self.basis_model(**{**{"random_state": R_STATE}, **hyper})
            scores = cross_val_score(model, X, y, cv=self.n_folds, scoring=self.scoring_metric)
            scores_list.append(scores.mean())

            if self.verbose:
                print(f"Hyperparameters {idx}: {scores.mean()} {self.scoring_metric} with a standard deviation of {scores.std()}")

        best_idx = getattr(np, f'arg{self.scoring_strategy}')(scores_list)
        return self.hyperparameters[best_idx]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        hyperparameters = self._cross_validation(X, y)
        self.model = self.basis_model(**{**{"random_state": R_STATE}, **hyperparameters}).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict_proba(X)