import asyncpg

from src.settings import DB_HOST, DB_USER, DB_PASS, DB_NAME, DB_PORT
import functools


def inject_db_connection():
    def wrapper(func):
        @functools.wraps(func)
        async def inner_wrapper(*args):
            conn = await asyncpg.connect(
                host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, port=DB_PORT
            )
            return await func(conn, *args)
        return inner_wrapper
    return wrapper



def upsert_to_db():
    pass
