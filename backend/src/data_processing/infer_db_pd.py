import pandas as pd
from config import config
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
    upt_df = pd.merge(df, new_data, how="outer")
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
    ratings = pd.DataFrame(columns=["username"])
    print(ratings.head())

    data = {"username": "jack", "Avatar": [5.6], "Shawshank": [8.0]}
    new_rating = pd.DataFrame(data=data)
    print(new_rating.head())

    new_df = pd.merge(ratings, new_rating, how="outer")
    print(new_df.head())
