import numpy as np
from typing import Tuple
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
        "num_leaves": trial.suggest_int("num_leaves", 15, 25, step=1),
        # Default: -1, use small to avoid overfitting
        "max_depth": trial.suggest_int("max_depth", 3, 8, step=1),
        # Default: 255, use small to avoid overfitting
        # "max_bin": trial.suggest_int("max_bin", 100, 255),
        # # Default: 20. Using small values since the metabase doesn't have many instances
        # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 30, step=5),
        # used to avoid overfitting since the metabase contains many columns
        # "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1, step=0.1),
        # "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        # "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        # "path_smooth": trial.suggest_float("path_smooth", 0, 50),
    }


class MetaModel():
    def __init__(
        self,
        param_map = default_param_map,
        n_folds: int = DEFAULT_N_FOLDS,
        verbose: bool = VERBOSE,
        n_trials: bool = DEFAULT_N_TRIALS,  # For optuna study
        random_state: int = R_STATE,
        select_k_features: Tuple[int, float] = None, # Quantity or proportion of features to be selected
        ):
        self.param_map = param_map
        self.n_folds = n_folds
        self.verbose = verbose
        self.n_trials = n_trials
        self.random_state = random_state
        self.select_k_features = select_k_features
        self.best_hyperparams = {}
        self.model = None
        self.feature_list = None

    def _objective(self, trial, features: pd.DataFrame, target: pd.Series):
        """Time series cross validation for finding the best hyperparam
        from the provided param map."""
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
        """Use optuna for automating the hyperparameter tuning step"""
        study = optuna.create_study(direction="minimize", study_name="Meta Model")
        func = lambda trial: self._objective(trial, features, target)
        study.optimize(func, n_trials=self.n_trials)
        return study.best_params
    
    def _get_n_most_important_features(self, model, n_features: int) -> list:
        importances = np.array(model.feature_importances_, dtype=float)
        imp_df =  pd.DataFrame({"name": model.feature_name_, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False)
        return list(imp_df.head(n_features)["name"])

    def _select_features(self, features: pd.DataFrame, target: pd.Series=None) -> pd.DataFrame:
        # If feature selection was already done
        if self.feature_list:
            return features[self.feature_list]

        # If feature selection is not needed use all features instead
        if not self.select_k_features or self.select_k_features==1:
            print("using all features")
            self.feature_list = list(features.columns)
            return features

        # If select_k_features is a proportion
        if self.select_k_features < 1:
            n_features = features.shape[1] * self.select_k_features
            self.select_k_features = int(np.ceil(n_features))  # round up

        # Train model and get most important features
        best_hyperparams = self._hyperparam_tuning(features, target)
        model = ltb.LGBMRegressor(**best_hyperparams).fit(features, target)
        self.feature_list = self._get_n_most_important_features(model, self.select_k_features)
        return features[self.feature_list]

    def _print(self, msg: str):
        if self.verbose:
            print(msg)

    def fit(self, features: pd.DataFrame, target: pd.Series):
        """Fit meta model and do hyperparameter tuning with optuna.
        The hyperparameter tuning is used only on first fit.
        """
        # Feature selection
        features = self._select_features(features, target)
        
        # do hyperparam tuning only on 1st training
        if not self.best_hyperparams:
            best_hyperparams = self._hyperparam_tuning(features, target)
            self.best_hyperparams = {"random_state": self.random_state, "verbose": -1, **best_hyperparams}
        self.model = ltb.LGBMRegressor(**self.best_hyperparams).fit(features, target)
        self._print("Finished meta model training")
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Make a prediction for the provided features"""
        features = self._select_features(features)
        return self.model.predict(features)
