import datetime

from pydantic import BaseModel



class MovieIn(BaseModel):
    title: str
    release_year: int
    genres: list[str]


class Movies(BaseModel):
    id: int
    title: str
    release_year: int
    genres: list[str]
    created_at: datetime.datetime
    updated_at: datetime.datetime


class MovieRatingIn(BaseModel):
    user: str
    movie: str
    rating: float


class MovieRating(MovieRatingIn):
    created_at: datetime.datetime
    updated_at: datetime.datetime
