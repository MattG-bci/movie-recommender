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
    ratings = _make_ratings_by_user(n_users=10, ratings_per_user=10)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.7, shuffle=False)

    assert len(train_ds) + len(val_ds) == 100
    assert len(train_ds) == 70
    assert len(val_ds) == 30


def test_construct_datasets_no_rating_leakage():
    ratings = _make_ratings_by_user(n_users=5, ratings_per_user=10)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.7, shuffle=False)

    train_ratings = {float(row[2]) for row in train_ds}
    val_ratings = {float(row[2]) for row in val_ds}

    assert train_ratings.isdisjoint(val_ratings), "No rating should appear in both sets"


def test_construct_datasets_every_user_in_train():
    ratings = _make_ratings_by_user(n_users=5, ratings_per_user=10)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.7, shuffle=False)

    train_user_ids = {int(row[0]) for row in train_ds}
    val_user_ids = {int(row[0]) for row in val_ds}

    assert train_user_ids == {0, 1, 2, 3, 4}
    assert val_user_ids == {0, 1, 2, 3, 4}


def test_construct_datasets_preserves_all_data():
    ratings = _make_ratings_by_user(n_users=5, ratings_per_user=10)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.7, shuffle=False)

    all_ratings = {float(row[2]) for row in train_ds} | {
        float(row[2]) for row in val_ds
    }
    expected_ratings = {float(i) for i in range(1, 51)}

    assert all_ratings == expected_ratings


def test_construct_datasets_every_user_has_at_least_one_train_rating():
    ratings = _make_ratings_by_user(n_users=5, ratings_per_user=2)
    train_ds, val_ds = construct_datasets(ratings, train_split=0.5, shuffle=False)

    train_user_ids = {int(row[0]) for row in train_ds}

    assert train_user_ids == {0, 1, 2, 3, 4}
    assert len(train_ds) >= 5
