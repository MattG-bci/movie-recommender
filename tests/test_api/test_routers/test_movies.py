import requests


def test_get_movies__default_page(api_url):
    out = requests.get(f"{api_url}/movies")
    body = out.json()
    assert out.status_code == 200
    assert body["total"] == 8
    assert body["limit"] == 50
    assert body["offset"] == 0
    assert len(body["items"]) == 8
    titles = [item["title"] for item in body["items"]]
    assert titles == sorted(titles)


def test_get_movies__respects_limit_and_offset(api_url):
    out = requests.get(f"{api_url}/movies", params={"limit": 2, "offset": 1})
    body = out.json()
    assert out.status_code == 200
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["offset"] == 1
    assert body["total"] == 8


def test_get_movies__search_is_case_insensitive(api_url):
    out = requests.get(f"{api_url}/movies", params={"search": "TEST_MOVIE1"})
    body = out.json()
    assert out.status_code == 200
    assert body["total"] == 1
    assert body["items"][0]["title"] == "test_movie1"


def test_get_movies__search_with_no_match_returns_empty_page(api_url):
    out = requests.get(f"{api_url}/movies", params={"search": "no_such_movie"})
    body = out.json()
    assert out.status_code == 200
    assert body["items"] == []
    assert body["total"] == 0


def test_get_movies__total_reflects_search_filter(api_url):
    out = requests.get(f"{api_url}/movies", params={"search": "test_movie1"})
    body = out.json()
    assert out.status_code == 200
    assert body["total"] == 1
    assert len(body["items"]) == 1


def test_get_movies__limit_above_max_returns_422(api_url):
    out = requests.get(f"{api_url}/movies", params={"limit": 101})
    assert out.status_code == 422


def test_get_movie__returns_full_metadata(api_url):
    out = requests.get(f"{api_url}/movies/1")
    body = out.json()
    assert out.status_code == 200
    assert body["title"] == "test_movie1"
    assert body["release_year"] == 2024
    assert body["director"] == "director1"
    assert body["country"] == "country1"
    assert body["genres"] == ["genre1", "genre2"]
    assert body["actors"] == ["actor1"]
    assert "poster_url" in body


def test_get_movie__unknown_id_returns_404(api_url):
    out = requests.get(f"{api_url}/movies/999999")
    assert out.status_code == 404


def test_get_movies__poster_url_is_null_without_api_key(api_url):
    # The test-api container has no OMDB_API_KEY configured, so posters must
    # degrade to null rather than the endpoint attempting a live OMDb call.
    out = requests.get(f"{api_url}/movies")
    body = out.json()
    assert out.status_code == 200
    assert all(item["poster_url"] is None for item in body["items"])
