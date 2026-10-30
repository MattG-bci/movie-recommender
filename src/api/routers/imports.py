from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from etl.generation.generate import create_uuid_for_user
from etl.letterboxd_import import (
    LetterboxdExportError,
    MAX_EXPORT_BYTES,
    parse_and_match_ratings,
)
from etl.sql_queries import (
    DatabaseConnector,
    fetch_movies_from_db,
    fetch_user_by_username,
    insert_usernames,
    upsert_movie_ratings,
)
from etl.ingestion import add_uuid_to_movie_ratings
from schemas.users import UserWithUUID


imports = APIRouter()


class LetterboxdImportResult(BaseModel):
    rated_rows: int
    imported_ratings: int
    unmatched_rows: int


@imports.post("/letterboxd/{username}", response_model=LetterboxdImportResult)
async def import_letterboxd_export(
    username: str, export: UploadFile = File(...)
) -> LetterboxdImportResult:
    """Import a user-selected Letterboxd export without handling their password."""
    username = username.lower().strip()
    if not username:
        raise HTTPException(
            status_code=400, detail="A Letterboxd username is required."
        )
    contents = await export.read(MAX_EXPORT_BYTES + 1)
    if len(contents) > MAX_EXPORT_BYTES:
        raise HTTPException(
            status_code=413, detail="The export is too large (maximum 15 MB)."
        )

    async with DatabaseConnector() as conn:
        user = await fetch_user_by_username(conn, username)
        if user is None:
            await insert_usernames(
                conn,
                [UserWithUUID(id=create_uuid_for_user(username), username=username)],
            )
            user = await fetch_user_by_username(conn, username)
        movies = await fetch_movies_from_db(conn)
        try:
            preview = parse_and_match_ratings(
                contents, movies, str(user.id), export.filename
            )
        except LetterboxdExportError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        ratings = add_uuid_to_movie_ratings(preview.ratings)
        await upsert_movie_ratings(conn, ratings)

    return LetterboxdImportResult(
        rated_rows=preview.rated_rows,
        imported_ratings=len(ratings),
        unmatched_rows=preview.unmatched_rows,
    )
