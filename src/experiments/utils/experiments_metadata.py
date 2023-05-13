import sys
sys.path.insert(0,'..')

# Models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


MODELS_METADATA = {
    "RandomForestClassifier": {
        "base_model_params": {
            "verbose": True,
            "basis_model": RandomForestClassifier,
            "hyperparameters": {"max_depth": 6}
        }},
    # "DecisionTreeClassifier": {
    #     "base_model_params": {
    #         "verbose": True,
    #         "basis_model": DecisionTreeClassifier,
    #         "hyperparameters": {"max_depth": 6}
    #     }},
    # "LogisticRegression": {
    #     "base_model_params": {
    #         "verbose": True,
    #         "basis_model": LogisticRegression,
    #         "hyperparameters": {}
    #     }},
    # "SVC": {
    #     "base_model_params": {
    #         "verbose": True,
    #         "basis_model": SVC,
    #         "hyperparameters": {"probability": True}
    #     }},
}

REG_MODELS_METADATA = {
    "RandomForestRegressor": {
        "base_model_params": {
            "verbose": True,
            "basis_model": RandomForestRegressor,
            "hyperparameters": {"max_depth": 6}
        }},
    # "DecisionTreeRegressor": {
    #     "base_model_params": {
    #         "verbose": True,
    #         "basis_model": DecisionTreeRegressor,
    #         "hyperparameters": {"max_depth": 6}
    #     }},
}

DATASETS_METADATA = {
    "powersupply": {
        "dataset_name": "powersupply",
        "class_col": "class",
        "base_model_type": "multiclass",
        "offline_phase_size": 5000,
        "base_train_size": 2000,
        "eta": 100,
        "step": 30,
        "target_delay": 500,
    },
    # "covtype": {
    #     "dataset_name": "covtype", # ERROR
    #     "class_col": "class",
    #     "base_model_type": "multiclass",
    #     "offline_phase_size": 50000,
    #     "base_train_size": 20000,
    #     "eta": 1000,
    #     "step": 300,
    #     "target_delay": 2000,
    # },
    "airlines": {
        "dataset_name": "airlines",
        "class_col": "Delay",
        "base_model_type": "binary_classification",
        "offline_phase_size": 50000,
        "base_train_size": 20000,
        "eta": 1000,
        "step": 300,
        "target_delay": 2000,
    },
    "electricity": {
        "dataset_name": "electricity",
        "class_col": "class",
        "base_model_type": "binary_classification",
        "offline_phase_size": 5000,
        "base_train_size": 2000,
        "eta": 100,
        "step": 30,
        "target_delay": 500,
    },
    # "poker": {
    #     "dataset_name": "poker-lsn", # ERROR
    #     "class_col": "class",
    #     "base_model_type": "multiclass",
    #     "offline_phase_size": 50000,
    #     "base_train_size": 20000,
    #     "eta": 1000,
    #     "step": 300,
    #     "target_delay": 2000,
    # },
    "rialto": {
        "dataset_name": "rialto",
        "class_col": "class",
        "base_model_type": "multiclass",
        "offline_phase_size": 5000,
        "base_train_size": 2000,
        "eta": 100,
        "step": 30,
        "target_delay": 500,
    },
}
