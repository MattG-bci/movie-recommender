import datetime

from pydantic import BaseModel


class Movies(BaseModel):
    id: int
    title: str
    release_year: int
    created_at: datetime.datetime
    updated_at: datetime.datetime


class MovieRating(BaseModel):
    id: int
    user_id: int
    movie_id: int
    rating: float
    created_at: datetime.datetime
    updated_at: datetime.datetime
