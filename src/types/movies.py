import datetime

from pydantic import BaseModel


class Movies(BaseModel):
    id: int
    title: str
    release_year: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
