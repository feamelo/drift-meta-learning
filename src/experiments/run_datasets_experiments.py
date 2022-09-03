from utils import run_experiment


# Metadata used for each dataset on meta learning algorithm
dataset_metadata = [
    {
        "dataset_name": "covtype",
        "class_col": "class",
        "base_model_type": "multiclass",
        "offline_phase_size": 50000,
        "base_train_size": 20000,
        "eta": 1000,
        "step": 300,
        "target_delay": 2000,
    },
    {
        "dataset_name": "airlines",
        "class_col": "Delay",
        "base_model_type": "binary_classification",
        "offline_phase_size": 50000,
        "base_train_size": 20000,
        "eta": 1000,
        "step": 300,
        "target_delay": 2000,
    },
    {
        "dataset_name": "electricity",
        "class_col": "class",
        "base_model_type": "binary_classification",
        "offline_phase_size": 5000,
        "base_train_size": 2000,
        "eta": 100,
        "step": 30,
        "target_delay": 500,
    },
    {
        "dataset_name": "powersupply",
        "class_col": "class",
        "base_model_type": "multiclass",
        "offline_phase_size": 5000,
        "base_train_size": 2000,
        "eta": 100,
        "step": 30,
        "target_delay": 500,
    },
    {
        "dataset_name": "poker-lsn",
        "class_col": "class",
        "base_model_type": "multiclass",
        "offline_phase_size": 50000,
        "base_train_size": 20000,
        "eta": 1000,
        "step": 300,
        "target_delay": 2000,
    },
    {
        "dataset_name": "rialto",
        "class_col": "class",
        "base_model_type": "multiclass",
        "offline_phase_size": 5000,
        "base_train_size": 2000,
        "eta": 100,
        "step": 30,
        "target_delay": 500,
    },
]

for metadata in dataset_metadata:
    run_experiment(metadata, include_drift=True, detail=metadata['dataset_name'])
    run_experiment(metadata, include_drift=False, detail=metadata['dataset_name'])
