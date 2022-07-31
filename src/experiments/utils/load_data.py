import pandas as pd
from scipy.io import arff


def load_electricity():
    file = open("../datasets/real/elec.arff", "r")
    data, _ = arff.loadarff(file)
    data_frame = pd.DataFrame(data)
    data_frame['weekday'] = pd.to_numeric(data_frame['day'])
    data_frame['class'] = data_frame['class'].str.decode("utf-8")
    return data_frame.drop(['date', 'period', 'day'], axis=1)


def load_dataset(dataset_name: str) -> pd.DataFrame:
    dataset_dict = {
        "electricity": load_electricity,
    }
    possible_datasets = list(dataset_dict.keys())
    if dataset_name not in possible_datasets:
        raise ValueError(f"Invalid dataset name, must be one of {possible_datasets}")
    return dataset_dict[dataset_name]()
