import datetime

from pydantic import BaseModel

class Users(BaseModel):
    id: int
    name: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
