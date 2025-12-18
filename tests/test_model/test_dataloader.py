from datetime import datetime

import torch

from model.dataloader import MoviesDataset, transform_rating_to_tensor
from schemas.movie import MovieRating


def test_singleton():
    dataloader_1 = MoviesDataset()
    dataloader_2 = MoviesDataset()
    assert dataloader_1 is dataloader_2


def test_transform_rating_to_tensor():
    mock_movie_rating = MovieRating(
        id=1,
        user_id=1,
        movie_id=2,
        rating=4.5,
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
    )
    res = transform_rating_to_tensor(mock_movie_rating)
    assert res == (
        torch.tensor(1, dtype=torch.float),
        torch.tensor(2, dtype=torch.float),
        torch.tensor(4.5, dtype=torch.float),
    )
