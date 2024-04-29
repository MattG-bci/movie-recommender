import pandas as pd
from config import config
import os


def init_db(name="pandas_db.parquet") -> None:
    df = pd.DataFrame(columns=["username"])
    df.to_parquet(os.path.join(config["db_dir"], name))
    return

if __name__ == "__main__":
    ratings = pd.DataFrame(columns=["username"])
    print(ratings.head())

    data = {"username": "jack", "Avatar": [5.6], "Shawshank": [8.0]}
    new_rating = pd.DataFrame(data=data)
    print(new_rating.head())

    new_df = pd.merge(ratings, new_rating, how="outer")
    print(new_df.head())
