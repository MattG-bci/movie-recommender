import random
from collections import defaultdict

import torch

from schemas.movie import MovieRatingWithId


class MoviesDataset(torch.utils.data.Dataset):
    def __init__(self, ratings: list[MovieRatingWithId]) -> None:
        self.ratings = [
            transform_movie_rating_with_id_to_tensor(rating) for rating in ratings
        ]

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> tuple[torch.tensor, ...]:
        return self.ratings[idx]


def construct_datasets(
    ratings: list[MovieRatingWithId], train_split: float = 0.7, shuffle: bool = True
) -> tuple[MoviesDataset, MoviesDataset]:
    assert 0.0 < train_split < 1.0, "train_split must be between 0 and 1"

    user_ratings: dict[int, list[MovieRatingWithId]] = defaultdict(list)
    for rating in ratings:
        user_ratings[rating.user_id].append(rating)

    user_ids = list(user_ratings.keys())
    if shuffle:
        random.shuffle(user_ids)

    train_user_count = int(train_split * len(user_ids))
    train_user_ids = set(user_ids[:train_user_count])

    train_ratings = [
        r for uid in user_ids if uid in train_user_ids for r in user_ratings[uid]
    ]
    val_ratings = [
        r for uid in user_ids if uid not in train_user_ids for r in user_ratings[uid]
    ]

    return MoviesDataset(train_ratings), MoviesDataset(val_ratings)


def transform_movie_rating_with_id_to_tensor(
    rating: MovieRatingWithId,
) -> tuple[torch.tensor, ...]:
    user_id_tensor = torch.tensor(rating.user_id, dtype=torch.long)
    movie_id_tensor = torch.tensor(rating.movie_id, dtype=torch.long)
    rating_value_tensor = torch.tensor(rating.rating, dtype=torch.float)
    return user_id_tensor, movie_id_tensor, rating_value_tensor
