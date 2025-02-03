import datetime

from pydantic import BaseModel

class User(BaseModel):
    name: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
