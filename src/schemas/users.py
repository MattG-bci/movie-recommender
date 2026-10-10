import datetime

from pydantic import BaseModel
from uuid import UUID


class UserIn(BaseModel):
    username: str

    # Uses memory address of an instance to hash it
    __hash__ = object.__hash__


class UserWithUUID(UserIn):
    id_uuid: UUID


class User(UserIn):
    id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
