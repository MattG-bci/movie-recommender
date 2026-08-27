from pydantic import BaseModel, confloat

import dspy
from schemas.movie import Movie


class RecommendationInput(BaseModel):
    username: str
    prompt: str
    image: dspy.Image | None = None
    exploration: confloat(strict=True, ge=0.0, le=1.0)
    n_cf_recommendations: int = 100
    top_k_recommendations: int = 10


class RecommendationOut(BaseModel):
    movie: Movie
    reason: str | None


class UserProfile(BaseModel):
    top_genres: list[str]
    top_actors: list[str]
    top_directors: list[str]
    top_movies: list[str]


class MovieCandidate(BaseModel):
    movie: Movie
    cf_score: float


class ImageSemantics(BaseModel):
    energy: str
    mood: str
    suggested_genres: list[str]
    avoid_genres: list[str]


class ExtractImageSemantics(dspy.Signature):
    image: dspy.Image = dspy.InputField(
        desc="Image describing the mood, vibe and current state which is used for reranking movies to help find more suitable candidates"
    )

    output: ImageSemantics = dspy.OutputField(
        desc="Semantics, mood and vibe extracted from an input image. Suggested movie genres to watch and to avoid included in the response"
    )


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
    exploration: confloat(strict=True, ge=0.0, le=1.0) = dspy.InputField(
        desc="The rate defining whether reranked should stick to the user's taste profile (0.0) or promote novelty and unseen genres, directors etc. (1.0)"
    )
    user_profile: UserProfile = dspy.InputField(
        desc="User's top genres, actors, directors, movies"
    )
    candidates: list[MovieCandidate] = dspy.InputField(
        desc="Candidate movies with id, title, metadata, cf_score"
    )
    image_semantics: ImageSemantics | None = dspy.InputField(
        desc="Image describing the mood, vibe and current state which is used for reranking movies to help find more suitable candidates"
    )

    ranked_ids: list[int] = dspy.OutputField(desc="Candidate movie_ids, best first")
    reasons: dict[int, str] = dspy.OutputField(desc="movie_id -> one-sentence reason")
