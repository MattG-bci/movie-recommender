"""Parsing and matching for a member-provided Letterboxd data export."""

from __future__ import annotations

import csv
import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath

from schemas.movie import Movie, MovieRatingIn


MAX_EXPORT_BYTES = 15 * 1024 * 1024


class LetterboxdExportError(ValueError):
    """The supplied file is not a safe, usable Letterboxd ratings export."""


@dataclass(frozen=True)
class ImportPreview:
    ratings: list[MovieRatingIn]
    rated_rows: int
    unmatched_rows: int


def normalise_title(title: str) -> str:
    """Make conservative title comparisons resilient to punctuation and spacing."""
    decomposed = unicodedata.normalize("NFKD", title)
    without_marks = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    return re.sub(r"[^\w]+", " ", without_marks.casefold()).strip()


def read_ratings_csv(
    contents: bytes, filename: str | None = None
) -> list[dict[str, str]]:
    if len(contents) > MAX_EXPORT_BYTES:
        raise LetterboxdExportError("The export is too large (maximum 15 MB).")

    if zipfile.is_zipfile(io.BytesIO(contents)):
        with zipfile.ZipFile(io.BytesIO(contents)) as archive:
            candidates = [
                entry
                for entry in archive.infolist()
                if PurePosixPath(entry.filename).name.casefold() == "ratings.csv"
            ]
            if len(candidates) != 1:
                raise LetterboxdExportError(
                    "The ZIP must contain exactly one ratings.csv file."
                )
            entry = candidates[0]
            if entry.file_size > MAX_EXPORT_BYTES or (
                entry.compress_size and entry.file_size / entry.compress_size > 100
            ):
                raise LetterboxdExportError(
                    "The ratings file is too large or highly compressed."
                )
            contents = archive.read(entry)
    elif filename and not filename.casefold().endswith(".csv"):
        raise LetterboxdExportError(
            "Choose a Letterboxd export ZIP or ratings.csv file."
        )

    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise LetterboxdExportError("ratings.csv must use UTF-8 encoding.") from error

    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or not rows[0] or "Name" not in rows[0] or "Rating" not in rows[0]:
        raise LetterboxdExportError(
            "The file does not look like a Letterboxd ratings.csv export."
        )
    return rows


def parse_and_match_ratings(
    contents: bytes, movies: list[Movie], user_id: str, filename: str | None = None
) -> ImportPreview:
    rows = read_ratings_csv(contents, filename)
    by_title_and_year = {
        (normalise_title(movie.title), movie.release_year): movie.id for movie in movies
    }
    by_title: dict[str, list[Movie]] = {}
    for movie in movies:
        by_title.setdefault(normalise_title(movie.title), []).append(movie)

    ratings: list[MovieRatingIn] = []
    rated_rows = 0
    unmatched_rows = 0
    for row in rows:
        rating = _parse_rating(row.get("Rating", ""))
        if rating is None:
            continue
        rated_rows += 1
        title = (row.get("Name") or "").strip()
        year = _parse_year(row.get("Year") or "")
        movie_id = (
            by_title_and_year.get((normalise_title(title), year)) if year else None
        )
        if movie_id is None:
            title_matches = by_title.get(normalise_title(title), [])
            movie_id = title_matches[0].id if len(title_matches) == 1 else None
        if movie_id is None:
            unmatched_rows += 1
            continue
        ratings.append(MovieRatingIn(user_id=user_id, movie_id=movie_id, rating=rating))

    # A title may occur more than once in an export; retain the final value.
    deduplicated = {rating.movie_id: rating for rating in ratings}
    return ImportPreview(
        ratings=list(deduplicated.values()),
        rated_rows=rated_rows,
        unmatched_rows=unmatched_rows,
    )


def _parse_rating(value: str) -> int | None:
    if not value.strip():
        return None
    try:
        stars = Decimal(value)
    except InvalidOperation:
        return None
    if not Decimal("0.5") <= stars <= Decimal("5"):
        return None
    return int(stars * 2)


def _parse_year(value: str) -> int | None:
    try:
        year = int(value)
    except ValueError:
        return None
    return year if 1888 <= year <= 3000 else None
