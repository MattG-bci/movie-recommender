import base64
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from api import routers as routers_module
from api.app import app
from schemas.movie import Movie
from schemas.recommendation import RecommendationOut
from api.routers.recommendations import decode_image


def _make_movie(id: int = 1, title: str = "Movie") -> Movie:
    return Movie(
        id=id,
        title=title,
        release_year=2024,
        genres=["Action"],
        director="Director",
        country="US",
        actors=["Actor"],
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_decode_image__none_returns_none():
    assert decode_image(None) is None


def test_decode_image__accepts_data_url_prefix():
    raw = b"fake-image-bytes"
    encoded = base64.b64encode(raw).decode()
    data_url = f"data:image/jpeg;base64,{encoded}"

    image = decode_image(data_url)

    assert image is not None


def test_decode_image__malformed_base64_raises_400():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        decode_image("not-valid-base64!!!")

    assert exc_info.value.status_code == 400


def test_post_recommendations__exploration_out_of_range_returns_422(client):
    response = client.post(
        "/recommendations",
        json={"username": "someone", "prompt": "something fun", "exploration": 1.5},
    )
    assert response.status_code == 422


def test_post_recommendations__unknown_username_returns_404(client, monkeypatch):
    async def _fake_recommend_movies(recommendation_input):
        raise ValueError(
            f"Profile for username={recommendation_input.username} has null fields: top_genres"
        )

    monkeypatch.setattr(
        routers_module.recommendations, "recommend_movies", _fake_recommend_movies
    )

    response = client.post(
        "/recommendations",
        json={"username": "no_such_user", "prompt": "something fun"},
    )
    assert response.status_code == 404


def test_post_recommendations__returns_match_score_and_reason(client, monkeypatch):
    async def _fake_recommend_movies(recommendation_input):
        return [
            RecommendationOut(
                movie=_make_movie(1), reason="great pick", match_score=9.0
            ),
            RecommendationOut(movie=_make_movie(2), reason=None, match_score=5.0),
        ]

    monkeypatch.setattr(
        routers_module.recommendations, "recommend_movies", _fake_recommend_movies
    )

    response = client.post(
        "/recommendations",
        json={"username": "testuser1", "prompt": "something fun"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body[0]["reason"] == "great pick"
    assert body[0]["match_score"] == pytest.approx(9.0)
    assert body[1]["reason"] is None
    assert body[1]["match_score"] == pytest.approx(5.0)


def test_post_recommendations__includes_poster_url_field(client, monkeypatch):
    async def _fake_recommend_movies(recommendation_input):
        return [RecommendationOut(movie=_make_movie(1), reason=None, match_score=9.0)]

    monkeypatch.setattr(
        routers_module.recommendations, "recommend_movies", _fake_recommend_movies
    )

    response = client.post(
        "/recommendations",
        json={"username": "testuser1", "prompt": "something fun"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "poster_url" in body[0]["movie"]


def _tmp_dir_snapshot() -> set[str]:
    return set(os.listdir(tempfile.gettempdir()))


def test_decode_image__creates_no_temp_file():
    raw = b"fake-image-bytes"
    encoded = base64.b64encode(raw).decode()

    before = _tmp_dir_snapshot()
    image = decode_image(encoded)
    after = _tmp_dir_snapshot()

    assert image is not None
    assert before == after


def test_decode_image__leaves_no_temp_file_after_successful_recommendation(
    client, monkeypatch
):
    async def _fake_recommend_movies(recommendation_input):
        return [RecommendationOut(movie=_make_movie(1), reason=None, match_score=9.0)]

    monkeypatch.setattr(
        routers_module.recommendations, "recommend_movies", _fake_recommend_movies
    )

    encoded = base64.b64encode(b"fake-image-bytes").decode()
    before = _tmp_dir_snapshot()
    response = client.post(
        "/recommendations",
        json={
            "username": "testuser1",
            "prompt": "something fun",
            "image_base64": encoded,
        },
    )
    after = _tmp_dir_snapshot()

    assert response.status_code == 200
    assert before == after


def test_decode_image__leaves_no_temp_file_when_recommend_movies_raises(
    client, monkeypatch
):
    async def _fake_recommend_movies(recommendation_input):
        raise ValueError(
            "Profile for username=no_such_user has null fields: top_genres"
        )

    monkeypatch.setattr(
        routers_module.recommendations, "recommend_movies", _fake_recommend_movies
    )

    encoded = base64.b64encode(b"fake-image-bytes").decode()
    before = _tmp_dir_snapshot()
    response = client.post(
        "/recommendations",
        json={
            "username": "no_such_user",
            "prompt": "something fun",
            "image_base64": encoded,
        },
    )
    after = _tmp_dir_snapshot()

    assert response.status_code == 404
    assert before == after


def test_decode_image__preserves_png_mime_from_data_url():
    raw = b"fake-image-bytes"
    encoded = base64.b64encode(raw).decode()
    data_url = f"data:image/png;base64,{encoded}"

    image = decode_image(data_url)

    assert image is not None
    assert image.url.startswith("data:image/png;base64,")


def test_decode_image__bare_base64_defaults_to_jpeg_mime():
    raw = b"fake-image-bytes"
    encoded = base64.b64encode(raw).decode()

    image = decode_image(encoded)

    assert image is not None
    assert image.url.startswith("data:image/jpeg;base64,")


def test_decode_image__tolerates_whitespace_in_base64_payload():
    raw = b"fake-image-bytes"
    encoded = base64.b64encode(raw).decode()
    with_whitespace = encoded[:4] + "\n" + encoded[4:] + " "

    image = decode_image(with_whitespace)

    assert image is not None
    assert image.url.startswith("data:image/jpeg;base64,")
