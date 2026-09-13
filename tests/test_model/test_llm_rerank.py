import pytest

from model import llm_rerank
from schemas.movie import Movie
from schemas.recommendation import MovieCandidate, UserProfile


def _make_movie(id: int, title: str = "Movie") -> Movie:
    return Movie(
        id=id,
        title=title,
        release_year=2024,
        genres=["Action"],
        director="Director",
        country="US",
        actors=["Actor"],
    )


def _make_candidates() -> list[MovieCandidate]:
    return [
        MovieCandidate(movie=_make_movie(1, "Movie A"), cf_score=9.0),
        MovieCandidate(movie=_make_movie(2, "Movie B"), cf_score=7.5),
        MovieCandidate(movie=_make_movie(3, "Movie C"), cf_score=5.0),
    ]


def _make_user_profile() -> UserProfile:
    return UserProfile(
        top_genres=["Action"],
        top_actors=["Actor"],
        top_directors=["Director"],
        top_movies=["Movie A"],
    )


class _FakePrediction:
    def __init__(self, ranked_ids: list[int], reasons: dict[int, str]) -> None:
        self.ranked_ids = ranked_ids
        self.reasons = reasons


class _FakeReranker:
    def __init__(self, prediction: _FakePrediction) -> None:
        self._prediction = prediction

    def __call__(self, **kwargs):
        return self._prediction


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch):
    """Avoid building a real dspy.LM (and needing an API key); dspy.context()
    accepts any object here because the reranker itself is stubbed out."""
    monkeypatch.setattr(llm_rerank, "get_lm", lambda: None)


@pytest.mark.asyncio
async def test_rerank_candidates__sets_match_score_from_cf_score(monkeypatch):
    prediction = _FakePrediction(ranked_ids=[1, 2], reasons={1: "great pick"})
    monkeypatch.setattr(llm_rerank, "MovieReranker", lambda: _FakeReranker(prediction))
    candidates = _make_candidates()

    results = await llm_rerank.rerank_candidates(
        user_profile=_make_user_profile(),
        prompt="something fun",
        exploration=0.3,
        candidates=candidates,
        k=2,
    )

    assert len(results) == 2
    assert results[0].movie.id == 1
    assert results[0].match_score == pytest.approx(9.0)
    assert results[0].reason == "great pick"
    assert results[1].movie.id == 2
    assert results[1].match_score == pytest.approx(7.5)


@pytest.mark.asyncio
async def test_rerank_candidates__filler_candidates_keep_match_score(monkeypatch):
    # The LLM only ranked one candidate; the remaining slot is filled from the
    # untouched candidates and must still carry its cf_score as match_score.
    prediction = _FakePrediction(ranked_ids=[1], reasons={1: "great pick"})
    monkeypatch.setattr(llm_rerank, "MovieReranker", lambda: _FakeReranker(prediction))
    candidates = _make_candidates()

    results = await llm_rerank.rerank_candidates(
        user_profile=_make_user_profile(),
        prompt="something fun",
        exploration=0.3,
        candidates=candidates,
        k=3,
    )

    assert len(results) == 3
    filler = results[1:]
    assert {r.movie.id for r in filler} == {2, 3}
    for r in filler:
        assert r.reason is None
        assert r.match_score is not None
