import datetime

from pydantic import BaseModel


class UserIn(BaseModel):
    username: str

    # Uses memory address of an instance to hash it
    __hash__ = object.__hash__


class User(UserIn):
    id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime


class UserProfile(BaseModel):
    user_id: int
    username: str
    top_genres: list[str]
    top_actors: list[str]
    top_directors: list[str]
    top_movies: list[str]
