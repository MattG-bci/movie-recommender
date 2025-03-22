import datetime

from pydantic import BaseModel


class UserIn(BaseModel):
    username: str

    # Uses memory address of an instance to hash it
    __hash__ = object.__hash__


class User(UserIn):
    created_at: datetime.datetime
    updated_at: datetime.datetime
