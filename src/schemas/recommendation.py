from pydantic import BaseModel, confloat

from schemas.movie import Movie


class RecommendationPrompt(BaseModel):
    prompt: str
    exploration: confloat(strict=True, ge=0.0, le=1.0)


class RecommendationOut(BaseModel):
    movie: Movie
    reason: str | None
    source: str
