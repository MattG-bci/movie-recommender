import datetime

from pydantic import BaseModel


class UserIn(BaseModel):
    username: str

    # This is a hack to make the UserIn class hashable
    __hash__ = object.__hash__


class User(UserIn):
    created_at: datetime.datetime
    updated_at: datetime.datetime
