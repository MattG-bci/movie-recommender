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
    id: UUID


class Movie(BaseModel):
    id: UUID | int
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
    user_id: UUID | int
    movie_id: UUID | int
    rating: int


class MovieRatingWithUUID(BaseModel):
    id: UUID
    user_id: UUID | int
    movie_id: UUID | int
    rating: int


class MovieRatingWithId(BaseModel):
    id: UUID | int
    user_id: UUID | int
    movie_id: UUID | int
    rating: int
