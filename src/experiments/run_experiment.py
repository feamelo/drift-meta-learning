from utils import MODELS_METADATA, DATASETS_METADATA
from utils import MtLRunner


dataset_metadata = DATASETS_METADATA["airlines"]
model_metadata = MODELS_METADATA["RandomForestClassifier"]

# Run experiment
MtLRunner(**model_metadata, **dataset_metadata, meta_label_metrics=["precision"]).run()
