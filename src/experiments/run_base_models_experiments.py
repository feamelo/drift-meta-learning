######################################################################
# Run experiments for different base models:
# RandomForestClassifier
# DecisionTreeClassifier
# LogisticRegression
# SVM
#
# Other metadata:
# Dataset: Electricity (binary clf)
# Meta label: kappa
######################################################################

import sys
sys.path.insert(0,'..')

from meta_learning import BaseModel
from utils import run_experiment

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# base_metadata = {
#     "dataset_name": "electricity",
#     "class_col": "class",
#     "base_model_type": "binary_classification",
#     "offline_phase_size": 5000,
#     "base_train_size": 2000,
#     "eta": 100,
#     "step": 30,
#     "target_delay": 500,
# }
base_metadata = {
    "dataset_name": "rialto",
    "class_col": "class",
    "base_model_type": "multiclass",
    "offline_phase_size": 5000,
    "base_train_size": 2000,
    "eta": 100,
    "step": 30,
    "target_delay": 500,
}
models_metadata = [
    # {"base_model": BaseModel(verbose=True, basis_model=RandomForestClassifier, hyperparameters={"max_depth": 6})},
    {"base_model": BaseModel(verbose=True, basis_model=DecisionTreeClassifier, hyperparameters={"max_depth": 6})},
    # {"base_model": BaseModel(verbose=True, basis_model=LogisticRegression, hyperparameters={})},
    # {"base_model": BaseModel(verbose=True, basis_model=SVC, hyperparameters={"probability": True})},
]
models_metadata = [{**metadata, **base_metadata} for metadata in models_metadata]

for metadata in models_metadata:
    try:
        run_experiment(metadata, include_drift=True, base_model=metadata["base_model"].basis_model.__name__, dataset=metadata["dataset_name"])
    except:
        print("asdf")
    try:
        run_experiment(metadata, include_drift=False, base_model=metadata["base_model"].basis_model.__name__, dataset=metadata["dataset_name"])
    except:
        print("asdf")
