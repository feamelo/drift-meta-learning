import numpy as np
import time
from utils import DriftContributionGenerator
from utils import DATASETS_METADATA, MODELS_METADATA


def get_window_size(metadata: dict) -> int:
    mtl_size = metadata["offline_phase_size"] - metadata["base_train_size"]
    eta = metadata["eta"]
    step = metadata["step"]
    window_size = (mtl_size - eta)/step
    return int(np.ceil(window_size))  # round up


# Run different datasets and base models
start = time.time()
for base_model in MODELS_METADATA.keys():
    for dataset_name, dataset_metadata in DATASETS_METADATA.items():
        for n_features in range(5, 101, 5):
            d_gen = DriftContributionGenerator(
                base_model=base_model,
                dataset_name=dataset_name,
                train_batch_size=get_window_size(dataset_metadata),
                meta_model_params={"select_k_features": n_features/100},
                output_filename=f"base_model: {base_model} - dataset: {dataset_name} - select_k_features: {n_features}%"
            )
            d_gen.run()
print(f"Finished - elapsed time: {time.time() - start}")
