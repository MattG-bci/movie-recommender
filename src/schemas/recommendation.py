from pydantic import BaseModel, confloat

import dspy
from schemas.movie import Movie


class RecommendationPrompt(BaseModel):
    prompt: str
    exploration: confloat(strict=True, ge=0.0, le=1.0)


class RecommendationOut(BaseModel):
    movie: Movie
    reason: str | None


class UserProfile(BaseModel):
    top_genres: list[str]
    top_actors: list[str]
    top_directors: list[str]
    top_movies: list[str]


class UserProfileWithID(UserProfile):
    user_id: int
    username: str


class MovieCandidate(BaseModel):
    movie: Movie
    cf_score: float


class RerankMovies(dspy.Signature):
    """
    Re-rank candidate movies for a user, balancing their established taste
    against their current request and the exploration setting.

    exploration ranges 0.0 (stick to the user's profile) to 1.0 (favour novelty:
    unfamiliar genres, directors, actors). Values between are a balance.

    Return ONLY movie_ids that appear in the provided candidates. For each pick,
    give a one-sentence reason; for exploratory picks, ground the reason in the
    shared actor/director that links it to the user's taste.
    """

    request: str = dspy.InputField(desc="The user's free-text recommendation request")
    exploration: float = dspy.InputField()
    user_profile: dict = dspy.InputField(
        desc="User's top genres, actors, directors, movies"
    )
    candidates: list[dict] = dspy.InputField(
        desc="Candidate movies with id, title, metadata, cf_score"
    )

    ranked_ids: list[int] = dspy.OutputField(desc="Candidate movie_ids, best first")
    reasons: dict[int, str] = dspy.OutputField(desc="movie_id -> one-sentence reason")
