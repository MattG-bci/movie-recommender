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


class MovieRating(BaseModel):
    username: str
    movie_name: str
    rating: float


class MovieRatingIn(BaseModel):
    user_id: int
    movie_id: int
    rating: float


class MovieRatingWithId(MovieRatingIn):
    id: int
