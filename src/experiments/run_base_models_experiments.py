import sys
sys.path.insert(0,'..')

from meta_learning import Model
from utils import run_experiment

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier


base_metadata = {
    "dataset_name": "electricity",
    "class_col": "class",
    "base_model_type": "binary_classification",
    "offline_phase_size": 5000,
    "base_train_size": 2000,
    "eta": 100,
    "step": 30,
    "target_delay": 500,
}
models_metadata = [
    # RandomForest was already included in datasets experiments - no need to run again
    # {"base_model": Model(verbose=True, basis_model=RandomForestClassifier, hyperparameters={"max_depth": 6})},
    {"base_model": Model(verbose=True, basis_model=DecisionTreeClassifier, hyperparameters={"max_depth": 6})},
    {"base_model": Model(verbose=True, basis_model=LogisticRegression, hyperparameters={})},
    {"base_model": Model(verbose=True, basis_model=SVC, hyperparameters={"probability": True})},
]
models_metadata = [{**metadata, **base_metadata} for metadata in models_metadata]

for metadata in models_metadata:
    run_experiment(metadata, include_drift=True, detail=metadata['base_model'].basis_model.__name__)
    run_experiment(metadata, include_drift=False, detail=metadata['base_model'].basis_model.__name__)
