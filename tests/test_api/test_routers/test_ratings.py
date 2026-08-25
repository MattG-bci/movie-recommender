def test_get_ratings__only_user(mock_client):
    out = mock_client.get("/ratings", params={"username": "testuser1"})
    assert len(out.json()) == 4


def test_get_ratings__only_movie(mock_client):
    out = mock_client.get("/ratings", params={"movie_name": "test_movie7"})
    assert len(out.json()) == 2


def test_get_ratings__both_user_and_movie(mock_client):
    out = mock_client.get(
        "/ratings", params={"username": "testuser1", "movie_name": "test_movie3"}
    )

    expected = [{"username": "testuser1", "movie_name": "test_movie3", "rating": 7}]
    assert out.json() == expected


def test_get_ratings__no_params(mock_client):
    out = mock_client.get("/ratings", params={})
    assert len(out.json()) == 14
