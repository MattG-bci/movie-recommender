import torch

from model.dataloader import construct_datasets, transform_rating_to_tensor
from schemas.movie import MovieRatingWithId


def test_transform_rating_to_tensor():
    mock_movie_rating = MovieRatingWithId(
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


def _make_ratings(n: int) -> list[MovieRatingWithId]:
    return [
        MovieRatingWithId(id=i, user_id=i % 3, movie_id=i % 5, rating=float(i))
        for i in range(1, n + 1)
    ]


def test_construct_datasets_split_sizes():
    ratings = _make_ratings(10)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.7, shuffle=False)

    assert len(train_ds) == 7
    assert len(val_ds) == 3
    assert len(train_ds) + len(val_ds) == 10


def test_construct_datasets_no_data_leakage():
    ratings = _make_ratings(10)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.7, shuffle=False)

    train_ratings = {float(row[2]) for row in train_ds}
    val_ratings = {float(row[2]) for row in val_ds}

    assert train_ratings.isdisjoint(
        val_ratings
    ), "Train and val sets should not overlap"


def test_construct_datasets_preserves_all_data():
    ratings = _make_ratings(10)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.7, shuffle=False)

    all_ratings = {float(row[2]) for row in train_ds} | {
        float(row[2]) for row in val_ds
    }
    expected_ratings = {float(i) for i in range(1, 11)}

    assert all_ratings == expected_ratings


def test_construct_datasets_different_split_ratios():
    ratings = _make_ratings(20)

    train_ds, val_ds = construct_datasets(ratings, train_split=0.5, shuffle=False)
    assert len(train_ds) == 10
    assert len(val_ds) == 10

    ratings = _make_ratings(20)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.9, shuffle=False)
    assert len(train_ds) == 18
    assert len(val_ds) == 2


def test_construct_datasets_order_preserved_without_shuffle():
    ratings = _make_ratings(10)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.7, shuffle=False)

    train_rating_values = [float(train_ds[i][2]) for i in range(len(train_ds))]
    val_rating_values = [float(val_ds[i][2]) for i in range(len(val_ds))]

    assert train_rating_values == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert val_rating_values == [8.0, 9.0, 10.0]
