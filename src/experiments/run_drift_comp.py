import numpy as np
from utils import DriftContributionGenerator
from utils import DATASETS_METADATA, MODELS_METADATA


def get_window_size(metadata: dict) -> int:
    mtl_size = metadata["offline_phase_size"] - metadata["base_train_size"]
    eta = metadata["eta"]
    step = metadata["step"]
    window_size = (mtl_size - eta)/step
    return int(np.ceil(window_size))  # round up

for base_model in MODELS_METADATA.keys():
    for dataset_name, dataset_metadata in DATASETS_METADATA.items():
        d_gen = DriftContributionGenerator(
            base_model=base_model,
            dataset_name=dataset_name,
            train_batch_size=get_window_size(dataset_metadata),
            n_models=1,
        )
        d_gen.run()
