import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.status import HTTP_404_NOT_FOUND

from etl.sql_queries import fetch_movie_ratings_from_db, fetch_usernames_from_db
from schemas.movie import MovieRating

app = FastAPI()


@app.get("/health")
async def get_healthcheck() -> dict[str, int]:
    return {"status": 200}


@app.get("/readiness")
async def get_readiness() -> dict[str, int]:
    return {"status": 200}


@app.get("/ratings")
async def get_ratings(movie_id: int | None = None) -> list[MovieRating]:
    movies = await fetch_movie_ratings_from_db()
    if movie_id is not None:
        movies = [movie for movie in movies if movie.movie_id == movie_id]
    return movies


@app.get("/ratings/{user}")
async def get_ratings_for_user(user: str) -> list[MovieRating]:
    movies = await fetch_movie_ratings_from_db()
    users = await fetch_usernames_from_db()
    map_user_name_to_id = {user.username: user.id for user in users}
    target_user_id = map_user_name_to_id.get(user)
    if target_user_id is None:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail=f"User '{user}' not found."
        )
    movies = [movie for movie in movies if movie.user_id == target_user_id]
    return movies


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
