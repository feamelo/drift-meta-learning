from utils import MODELS_METADATA, DATASETS_METADATA
from utils import run_experiment


dataset_metadata = DATASETS_METADATA["airlines"]
model_metadata = MODELS_METADATA["LogisticRegression"]
experiment_metadata = {**model_metadata, **dataset_metadata}

params = {
    "data": experiment_metadata,
    "base_model": experiment_metadata["base_model"].basis_model.__name__,
    "dataset": experiment_metadata["dataset_name"],
}
# run_experiment(include_drift=True, **params)
run_experiment(include_drift=False, **params)
