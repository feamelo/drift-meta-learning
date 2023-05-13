from utils import MODELS_METADATA, DATASETS_METADATA, REG_MODELS_METADATA
from utils import MtLRunner
import os


dataset_metadata = {
    "class_col": "target",
    "offline_phase_size": 3000,
    "base_train_size": 1500,
    "eta": 100,
    "step": 30,
    "target_delay": 300,
}
clf_metrics = ["kappa", "precision", "recall", "f1-score"]
reg_metrics = ["r2", "mse", "std"]
bin_clf_metadata = {"base_model_type": "binary_classification", **dataset_metadata}
multiclass_metadata = {"base_model_type": "multiclass", **dataset_metadata}
reg_metadata = {"base_model_type": "regression", **dataset_metadata}


def run_real_datasets_experiments():
    for _, dataset_metadata in DATASETS_METADATA.items():
        for _, model_metadata in MODELS_METADATA.items():
            MtLRunner(**model_metadata, **dataset_metadata, meta_label_metrics=clf_metrics).run()


def run_synthetic_datasets_experiments(data_type="clf"):
    if data_type == "clf":
        datasets = [file for file in os.listdir("../datasets/synthetic/") if ("csv" in file and "friedman" not in file)]
        metrics = clf_metrics
    else:
        datasets = ["gradual_friedman.csv"]
        metrics = reg_metrics

    for dataset in datasets:
        for _, model_metadata in MODELS_METADATA.items():
            dataset_metadata["dataset_name"] = dataset
            print(dataset_metadata)
            MtLRunner(**model_metadata, **dataset_metadata, meta_label_metrics=metrics).run()

# run_synthetic_datasets_experiments()
