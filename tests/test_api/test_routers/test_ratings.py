import requests


def test_get_ratings__only_user(api_url):
    out = requests.get(f"{api_url}/ratings", params={"username": "testuser1"})
    assert len(out.json()) == 4


def test_get_ratings__only_movie(api_url):
    out = requests.get(f"{api_url}/ratings", params={"movie_name": "test_movie7"})
    assert len(out.json()) == 2


def test_get_ratings__both_user_and_movie(api_url):
    out = requests.get(
        f"{api_url}/ratings",
        params={"username": "testuser1", "movie_name": "test_movie3"},
    )

    expected = [{"username": "testuser1", "movie_name": "test_movie3", "rating": 7}]
    assert out.json() == expected


def test_get_ratings__no_params(api_url):
    out = requests.get(f"{api_url}/ratings", params={})
    assert len(out.json()) == 14


def test_post_ratings__scrapes_and_saves_ratings_for_existing_user(
    api_url, delete_user_and_ratings
):
    username = "api-ratings-user"
    try:
        user_response = requests.post(f"{api_url}/users/{username}")
        assert user_response.status_code == 200

        out = requests.post(f"{api_url}/ratings/{username}")
        assert out.status_code == 200

        ratings = requests.get(f"{api_url}/ratings", params={"username": username})
        assert ratings.status_code == 200
        assert ratings.json() == [
            {"username": username, "movie_name": "test_movie1", "rating": 10},
            {"username": username, "movie_name": "test_movie2", "rating": 7},
        ]
    finally:
        delete_user_and_ratings(username)
