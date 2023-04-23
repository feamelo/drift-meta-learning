from utils import MODELS_METADATA, DATASETS_METADATA
from utils import MtLRunner
import os


dataset_metadata = {
    "class_col": "target",
    "base_model_type": "multiclass",
    "offline_phase_size": 3000,
    "base_train_size": 1500,
    "eta": 100,
    "step": 30,
    "target_delay": 300,
}
metrics = ["kappa", "precision", "recall", "f1-score"]

# # Run real data experiments
# dataset_metadata = DATASETS_METADATA["airlines"]
# model_metadata = MODELS_METADATA["RandomForestClassifier"]
# MtLRunner(**model_metadata, **dataset_metadata, meta_label_metrics=["precision"]).run()

# Run synthetic data experiments
datasets = [file for file in os.listdir("../datasets/synthetic/") if "csv" in file]
for dataset in datasets:
    for _, model_metadata in MODELS_METADATA.items():
        dataset_metadata["dataset_name"] = dataset
        print(dataset_metadata)
        MtLRunner(**model_metadata, **dataset_metadata, meta_label_metrics=metrics).run()
