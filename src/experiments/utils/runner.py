from utils import MetabaseGenerator
import pandas as pd
import time


# Macros
TIME_ELAPSED_FILE = "results/elapsed_time.csv"

def run_experiment(data, include_drift=True, detail=""):
    """Generate metabase for the specified data.
    Measures the elapsed time for running the meta learning algorithm
    and save it to a csv file for later comparison.
    """
    print(f"Generating metabase for {detail} with{'out' if not include_drift else ''} drift metrics")
    # Load existing time elapsed dataframe
    time_df = pd.read_csv(TIME_ELAPSED_FILE)
    # Start timer
    start = time.time()
    # Run experiment
    mb_gen_with_drift = MetabaseGenerator(**data, include_drift_metrics=include_drift)
    mb_gen_with_drift.run()
    # Append time elapsed data to dataframe
    time_df.loc[len(time_df)] ={
        "experiment": "meta_labels",
        "detail": detail,
        "include_drift": include_drift,
        "elapsed_time": time.time() - start,
    }
    # Save execution data
    time_df.to_csv(TIME_ELAPSED_FILE, index=False)
