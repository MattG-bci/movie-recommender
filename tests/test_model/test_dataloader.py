import torch

from model.dataloader import transform_rating_to_tensor
from schemas.movie import MovieRating


def test_transform_rating_to_tensor():
    mock_movie_rating = MovieRating(
        id=1,
        user_id=1,
        movie_id=2,
        rating=4.5,
    )
    res = transform_rating_to_tensor(mock_movie_rating)
    assert res == (
        torch.tensor(1, dtype=torch.float),
        torch.tensor(2, dtype=torch.float),
        torch.tensor(4.5, dtype=torch.float),
    )
