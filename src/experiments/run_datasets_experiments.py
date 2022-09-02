from utils import dataset_metadata
from metabase_generator import MetabaseGenerator

for dataset in dataset_metadata:
    try:
        mb_gen_with_drift = MetabaseGenerator(**dataset, include_drift_metrics=True)
        mb_gen_with_drift.run()

        mb_gen_no_drift = MetabaseGenerator(**dataset, include_drift_metrics=False)
        mb_gen_no_drift.run()
    except:
        print(f"ERROR for {dataset['dataset_name']}")
