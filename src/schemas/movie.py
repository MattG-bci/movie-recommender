from pydantic import BaseModel
from uuid import UUID


class MovieIn(BaseModel):
    title: str
    release_year: int
    director: str
    country: str
    actors: list[str]
    genres: list[str]
    description: str | None = None


class MovieWithUUID(MovieIn):
    id_uuid: UUID


class Movie(BaseModel):
    id: int
    id_uuid: UUID | None = None
    title: str
    release_year: int
    genres: list[str]
    director: str
    country: str
    actors: list[str]
    description: str | None = None


class MovieOut(Movie):
    poster_url: str | None = None


class MoviePage(BaseModel):
    items: list[MovieOut]
    total: int
    limit: int
    offset: int


class MovieRating(BaseModel):
    username: str
    movie_name: str
    rating: int


class MovieRatingIn(BaseModel):
    user_id: int
    user_uuid: UUID
    movie_id: int
    movie_uuid: UUID
    rating: int


class MovieRatingWithUUID(BaseModel):
    id_uuid: UUID
    user_id: int
    movie_id: int
    rating: int


class MovieRatingWithId(BaseModel):
    id: int
    user_id: int
    movie_id: int
    rating: int
