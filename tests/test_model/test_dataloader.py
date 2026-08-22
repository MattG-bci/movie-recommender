import torch

from model.dataloader import (
    construct_datasets,
    transform_movie_rating_with_id_to_tensor,
)
from schemas.movie import MovieRatingWithId


def test_transform_rating_to_tensor():
    mock_movie_rating = MovieRatingWithId(
        id=1,
        user_id=1,
        movie_id=2,
        rating=4.5,
    )
    res = transform_movie_rating_with_id_to_tensor(mock_movie_rating)
    assert res == (
        torch.tensor(1, dtype=torch.float),
        torch.tensor(2, dtype=torch.float),
        torch.tensor(4.5, dtype=torch.float),
    )


def _make_ratings_by_user(
    n_users: int, ratings_per_user: int
) -> list[MovieRatingWithId]:
    ratings = []
    rating_id = 1
    for user_id in range(n_users):
        for j in range(ratings_per_user):
            ratings.append(
                MovieRatingWithId(
                    id=rating_id,
                    user_id=user_id,
                    movie_id=rating_id % 5,
                    rating=float(rating_id),
                )
            )
            rating_id += 1
    return ratings


def test_construct_datasets_split_sizes():
    ratings = _make_ratings_by_user(n_users=10, ratings_per_user=3)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.7, shuffle=False)

    assert len(train_ds) == 21
    assert len(val_ds) == 9
    assert len(train_ds) + len(val_ds) == 30


def test_construct_datasets_no_user_leakage():
    ratings = _make_ratings_by_user(n_users=10, ratings_per_user=5)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.7, shuffle=False)

    train_user_ids = {int(row[0]) for row in train_ds}
    val_user_ids = {int(row[0]) for row in val_ds}

    assert train_user_ids.isdisjoint(
        val_user_ids
    ), "No user should appear in both train and val sets"


def test_construct_datasets_preserves_all_data():
    ratings = _make_ratings_by_user(n_users=10, ratings_per_user=3)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.7, shuffle=False)

    all_ratings = {float(row[2]) for row in train_ds} | {
        float(row[2]) for row in val_ds
    }
    expected_ratings = {float(i) for i in range(1, 31)}

    assert all_ratings == expected_ratings


def test_construct_datasets_different_split_ratios():
    ratings = _make_ratings_by_user(n_users=10, ratings_per_user=2)

    train_ds, val_ds = construct_datasets(ratings, train_split=0.5, shuffle=False)
    assert len(train_ds) == 10
    assert len(val_ds) == 10

    ratings = _make_ratings_by_user(n_users=10, ratings_per_user=2)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.9, shuffle=False)
    assert len(train_ds) == 18
    assert len(val_ds) == 2


def test_construct_datasets_order_preserved_without_shuffle():
    ratings = _make_ratings_by_user(n_users=5, ratings_per_user=2)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.6, shuffle=False)

    train_rating_values = [float(train_ds[i][2]) for i in range(len(train_ds))]
    val_rating_values = [float(val_ds[i][2]) for i in range(len(val_ds))]

    assert train_rating_values == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert val_rating_values == [7.0, 8.0, 9.0, 10.0]
