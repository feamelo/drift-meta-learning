from utils import MODELS_METADATA, DATASETS_METADATA
from utils import MtLRunner


dataset_metadata = DATASETS_METADATA["electricity"]
model_metadata = MODELS_METADATA["LogisticRegression"]

# Run experiment
MtLRunner(**model_metadata, **dataset_metadata).run()
