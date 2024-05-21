import pandas as pd
from utils.config import config
import os
from typing import List


def init_db(name="pandas_db.parquet") -> None:
    df = pd.DataFrame(columns=["username"])
    df.to_parquet(os.path.join(config["db_dir"], config["db_name"]))
    return

def load_db() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(config["db_dir"], config["db_name"]), engine="pyarrow")

def append_data_samples(data: dict, df: pd.DataFrame) -> pd.DataFrame:
    new_data = pd.DataFrame(data=data)
    upt_df = pd.merge(df, new_data, how="outer", sort=False)
    return upt_df

def save_db(df: pd.DataFrame) -> None:
    df.to_parquet(os.path.join(config["db_dir"], config["db_name"]))
    return

def transform_data(username: str, data: List) -> dict:
    data_dict = dict(data)
    for key, value in data_dict.items():
        data_dict[key] = [value]
    data_dict["username"] = username
    return data_dict




if __name__ == "__main__":
    pass