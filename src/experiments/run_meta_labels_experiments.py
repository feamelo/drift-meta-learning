######################################################################
# Run experiments for different base labels:
# kappa
# F1
# recall
# precision
# AUC
#
# Other metadata:
# Dataset: Electricity (binary clf)
# Base model: RandomForestClassifier
######################################################################

from utils import run_experiment
from meta_learning import evaluator


# Not running for kappa since this is already executed in datasets experiment
metrics = [metric for metric in evaluator.binary_clf_metrics if metric != "kappa"]
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
metric_metadata = [{"meta_label_metric": metric, **base_metadata} for metric in metrics]


for metadata in metric_metadata:
    run_experiment(metadata, include_drift=True, detail=metadata['meta_label_metric'])
    run_experiment(metadata, include_drift=False, detail=metadata['meta_label_metric'])
