import polars as pl
from utils.config import config
import os
from typing import List


def load_db() -> pl.DataFrame:
    return pl.read_parquet(
        os.path.join(config["db_dir"], config["db_name"])
    )


def append_data_samples(data: dict, df: pl.DataFrame) -> pl.DataFrame:
    new_data = pl.DataFrame(data=data)
    upt_df = df.join(new_data, how="cross")
    return upt_df


def save_db(df: pl.DataFrame) -> None:
    df.write_parquet(os.path.join(config["db_dir"], config["db_name"]))
    return


def transform_data(username: str, data: List) -> dict:
    data_dict = dict(data)
    for key, value in data_dict.items():
        data_dict[key] = [value]
    data_dict["username"] = username
    return data_dict


if __name__ == "__main__":
    pass
