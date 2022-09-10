from utils import MODELS_METADATA, DATASETS_METADATA
from utils import run_experiment


dataset_metadata = DATASETS_METADATA["powersupply"]
experiment_metadata = [{**metadata, **dataset_metadata} for metadata in MODELS_METADATA.values()]

for metadata in experiment_metadata:
    params = {
        "data": metadata,
        "base_model": metadata["base_model"].basis_model.__name__,
        "dataset": metadata["dataset_name"],
    }
    run_experiment(include_drift=True, **params)
    run_experiment(include_drift=False, **params)
