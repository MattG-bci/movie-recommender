import datetime

from pydantic import BaseModel


class UserIn(BaseModel):
    username: str


class User(UserIn):
    created_at: datetime.datetime
    updated_at: datetime.datetime
