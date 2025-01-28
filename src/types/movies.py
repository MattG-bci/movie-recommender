from pydantic import BaseModel


class Movies(BaseModel):
    id: int
    title: str
    year: int
    genre: str
