import datetime

from pydantic import BaseModel


class MovieIn(BaseModel):
    title: str
    release_year: int
    director: str
    country: str
    actors: list[str]
    genres: list[str]


class Movie(BaseModel):
    id: int
    title: str
    release_year: int
    genres: list[str]
    director: str
    country: str
    actors: list[str]
    created_at: datetime.datetime
    updated_at: datetime.datetime


class MovieRatingIn(BaseModel):
    user_id: int
    movie_id: int
    rating: float


class MovieRating(MovieRatingIn):
    id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
