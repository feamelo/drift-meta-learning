import pandas as pd
from scipy.io import arff


def _get_cat_cols(data_frame: pd.DataFrame) -> list:
    cols = data_frame.columns.to_series().groupby(data_frame.dtypes).groups
    return list({k.name: v for k, v in cols.items()}["object"])


def _decode_df_cols(data_frame: pd.DataFrame) -> pd.DataFrame:
    cols = _get_cat_cols(data_frame)
    for col in cols:
        data_frame[col] = data_frame[col].str.decode("utf-8")
    return data_frame


def _load_generic(name: str) -> pd.DataFrame:
    file = open(f"../datasets/real/{name}.arff", "r", encoding="utf-8")
    data, _ = arff.loadarff(file)
    data_frame = pd.DataFrame(data)
    return _decode_df_cols(data_frame)


def _load_covertype() -> pd.DataFrame:
    return _load_generic("covtype")


def _load_powersupply() -> pd.DataFrame:
    return _load_generic("powersupply")


def _load_electricity() -> pd.DataFrame:
    file = open("../datasets/real/elec.arff", "r", encoding="utf-8")
    data, _ = arff.loadarff(file)
    data_frame = pd.DataFrame(data)
    data_frame['weekday'] = pd.to_numeric(data_frame['day'])
    data_frame['class'] = data_frame['class'].str.decode("utf-8")
    return data_frame.drop(['date', 'period', 'day'], axis=1)


def load_dataset(dataset_name: str) -> pd.DataFrame:
    dataset_dict = {
        "electricity": _load_electricity,
        "covertype": _load_covertype,
        "powersupply": _load_powersupply,
    }
    possible_datasets = list(dataset_dict.keys())
    if dataset_name not in possible_datasets:
        raise ValueError(f"Invalid dataset name, must be one of {possible_datasets}")
    return dataset_dict[dataset_name]()
