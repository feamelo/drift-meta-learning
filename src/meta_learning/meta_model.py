import numpy as np
import pandas as pd
import lightgbm as ltb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# hyperparam optimization
from optuna.integration import LightGBMPruningCallback
import optuna


# Macros
DEFAULT_N_FOLDS = 5
VERBOSE = True
R_STATE = 2022
DEFAULT_N_TRIALS = 10


def default_param_map(trial):
    """Param map to be used for optuna optimization.
    These parameters were chosen based on lightGBM documentation reccomendations
    for overfitting handling.
    """
    return {
        # Default: 31, use small to avoid overfitting
        "num_leaves": trial.suggest_int("num_leaves", 15, 30, step=20),
        # Default: -1, use small to avoid overfitting
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        # Default: 255, use small to avoid overfitting
        "max_bin": trial.suggest_int("max_bin", 100, 255),
        # Default: 20. Using small values since the metabase doesn't have many instances
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 30, step=5),
        # used to avoid overfitting since the metabase contains many columns
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1, step=0.1),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "path_smooth": trial.suggest_float("path_smooth", 0, 50),
    }


class MetaModel():
    def __init__(
        self,
        param_map = default_param_map,
        n_folds: int = DEFAULT_N_FOLDS,
        verbose: bool = VERBOSE,
        n_trials: bool = DEFAULT_N_TRIALS,  # For optuna study
        ):
        self.param_map = param_map
        self.n_folds = n_folds
        self.verbose = verbose
        self.n_trials = n_trials
        self.best_hyperparams = {}
        self.model = None

    def _objective(self, trial, features: pd.DataFrame, target: pd.Series):
        cross_val = TimeSeriesSplit(n_splits=5)
        cv_scores = np.empty(5)
        hyperparams = self.param_map(trial)
        for idx, (train_idx, test_idx) in enumerate(cross_val.split(features, target)):
            x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = target[train_idx], target[test_idx]

            model = ltb.LGBMRegressor(verbose=-1, random_state=R_STATE, **hyperparams)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_test, y_test)],
                eval_metric="mse",
                early_stopping_rounds=100,
                callbacks=[LightGBMPruningCallback(trial, "l2")],
            )
            preds = model.predict(x_test)
            cv_scores[idx] = mean_squared_error(y_test, preds)
        return np.mean(cv_scores)

    def _hyperparam_tuning(self, features: pd.DataFrame, target: pd.Series) -> dict:
        study = optuna.create_study(direction="minimize", study_name="Meta Model")
        func = lambda trial: self._objective(trial, features, target)
        study.optimize(func, n_trials=self.n_trials)
        return study.best_params

    def _print(self, msg: str):
        if self.verbose:
            print(msg)

    def fit(self, features: pd.DataFrame, target: pd.Series):
        """Fit meta model and do hyperparameter tuning with optuna.
        The hyperparameter tuning is used only on first fit.
        """
        if not self.best_hyperparams:  # do hyperparam tuning only on 1st training
            self._print("Starting hyperparam tuning")
            best_hyperparams = self._hyperparam_tuning(features, target)
            self._print(f"Best hyperparams: {best_hyperparams}")
            self.best_hyperparams = {"random_state": R_STATE, "verbose": -1, **best_hyperparams}
        self._print("Training meta model")
        self.model = ltb.LGBMRegressor(**self.best_hyperparams).fit(features, target)
        self._print("Finished meta model training")
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Make a prediction for the provided features"""
        return self.model.predict(features)
