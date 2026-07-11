from pydantic import BaseModel, confloat

from schemas.movie import Movie


class RecommendationPrompt(BaseModel):
    prompt: str
    exploration: confloat(strict=True, ge=0.0, le=1.0)


class RecommendationOut(BaseModel):
    movie: Movie
    reason: str | None


class UserProfile(BaseModel):
    user_id: int
    username: str
    top_genres: list[str]
    top_actors: list[str]
    top_directors: list[str]
    top_movies: list[str]


class MovieCandidate(BaseModel):
    movie: Movie
    cf_score: float
