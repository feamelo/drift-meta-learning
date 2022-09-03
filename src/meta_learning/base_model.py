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
DEFAULT_SCORING_STRATEGY = "max"
SCORING_METRICS = sklearn.metrics.SCORERS.keys()
DEFAULT_METRIC = "precision"
SCORING_STRATEGIES = ["max", "min"]
DEFAULT_CROSS_VAL_TYPE = "time_series"
CROSS_VAL_TYPES = ["time_series", "kfold"]
VERBOSE = False
R_STATE = 2022


class BaseModel():
    def __init__(
        self,
        hyperparameters: Tuple[list, dict] = None,
        basis_model = DEFAULT_MODEL,
        n_folds: int = DEFAULT_N_FOLDS,
        scoring_metric: str = DEFAULT_METRIC,
        scoring_strategy: str = DEFAULT_SCORING_STRATEGY,
        cross_val_type: str = DEFAULT_CROSS_VAL_TYPE,
        verbose: bool = VERBOSE,
        ):
        if scoring_metric not in SCORING_METRICS:
            raise Exception(f"Invalid scoring metric, must be one of: {SCORING_METRICS}")
        if scoring_strategy not in SCORING_STRATEGIES:
            raise Exception(f"Invalid scoring strategy, must be one of: {SCORING_STRATEGIES}")
        if cross_val_type not in CROSS_VAL_TYPES:
            raise Exception(f"Invalid cross validation type, must be one of: {CROSS_VAL_TYPES}")

        self.basis_model = basis_model
        self.hyperparameters = hyperparameters
        self.n_folds = n_folds
        self.scoring_metric = scoring_metric
        self.scoring_strategy = scoring_strategy
        self.cross_val_type = cross_val_type
        self.verbose = verbose
        self.best_hyperparams = {}
        self.model = None

    def _kfold_cv(self, features: pd.DataFrame, target: pd.Series, model) -> float:
        return cross_val_score(
            model,
            features,
            target,
            cv=self.n_folds,
            scoring=self.scoring_metric
        )

    def _time_series_cv(self, features: pd.DataFrame, target: pd.Series, model) -> float:
        cross_val = TimeSeriesSplit(n_splits=self.n_folds).split(features)
        return cross_val_score(model, features, target, cv=cross_val, scoring=self.scoring_metric)

    def _cross_validation(self, features: pd.DataFrame, target: pd.Series) -> dict:
        if isinstance(self.hyperparameters, dict):
            return self.hyperparameters
        if not self.hyperparameters:
            return {}

        scores_list = []
        for idx, hyper in enumerate(self.hyperparameters):
            model = self.basis_model(**{**{"random_state": R_STATE}, **hyper})

            if self.cross_val_type == "kfold":
                scores = self._kfold_cv(features, target, model)
            else:
                scores = self._time_series_cv(features, target, model)
            scores_list.append(scores.mean())

            if self.verbose:
                print(f"Hyperparameters {idx}: {scores.mean()} {self.scoring_metric} \
                    with a standard deviation of {scores.std()}")

        best_idx = getattr(np, f'arg{self.scoring_strategy}')(scores_list)
        return self.hyperparameters[best_idx]

    def fit(self, features: pd.DataFrame, target: pd.Series):
        best_hyperparams = self._cross_validation(features, target)
        self.best_hyperparams = {**{"random_state": R_STATE}, **best_hyperparams}
        self.model = self.basis_model(**self.best_hyperparams).fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return self.model.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        return self.model.predict_proba(features)

    def partial_fit(self, features: pd.DataFrame, target: pd.Series):
        """Only available for lightGBM model"""
        if not hasattr(self, 'model'):
            self.fit(features, target)
        else:
            init_model = self.model.booster_
            self.model = self.basis_model(**self.best_hyperparams).fit(features, target, init_model=init_model)
