import random

import torch

from schemas.movie import MovieRating


class MoviesDataset(torch.utils.data.Dataset):
    def __init__(self, ratings: list[MovieRating]) -> None:
        self.ratings = [transform_rating_to_tensor(rating) for rating in ratings]

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> tuple[torch.tensor, ...]:
        return self.ratings[idx]


def construct_datasets_for_train_eval(
    ratings: list[MovieRating], train_split: float = 0.7, shuffle: bool = True
) -> tuple[MoviesDataset, MoviesDataset]:
    assert 0.0 < train_split < 1.0, "train_split must be between 0 and 1"
    if shuffle:
        random.shuffle(ratings)
    train_size = int(train_split * len(ratings))
    train_ratings = ratings[:train_size]
    eval_ratings = ratings[train_size:]
    return MoviesDataset(train_ratings), MoviesDataset(eval_ratings)


def transform_rating_to_tensor(rating: MovieRating) -> tuple[torch.tensor, ...]:
    user_id_tensor = torch.tensor(rating.user_id, dtype=torch.long)
    movie_id_tensor = torch.tensor(rating.movie_id, dtype=torch.long)
    rating_value_tensor = torch.tensor(rating.rating, dtype=torch.float)
    return user_id_tensor, movie_id_tensor, rating_value_tensor
