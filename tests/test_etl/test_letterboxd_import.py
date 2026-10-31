import io
import zipfile
from uuid import uuid4

import pytest

from etl.letterboxd_import import LetterboxdExportError, parse_and_match_ratings
from schemas.movie import Movie


def _movie(title: str, year: int) -> Movie:
    return Movie(
        id=uuid4(),
        title=title,
        release_year=year,
        genres=[],
        director="Director",
        country="US",
        actors=[],
    )


def test_parse_and_match_ratings__reads_zip_and_converts_half_stars() -> None:
    inception = _movie("Inception", 2010)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as export:
        export.writestr(
            "ratings.csv",
            "Date,Name,Year,Letterboxd URI,Rating\n"
            "2026-01-01,Inception,2010,https://letterboxd.com/film/inception/,4.5\n"
            "2026-01-02,Missing Film,2020,https://letterboxd.com/film/missing-film/,3\n",
        )

    preview = parse_and_match_ratings(
        archive.getvalue(), [inception], str(uuid4()), "letterboxd-export.zip"
    )

    assert preview.rated_rows == 2
    assert preview.unmatched_rows == 1
    assert len(preview.ratings) == 1
    assert preview.ratings[0].movie_id == inception.id
    assert preview.ratings[0].rating == 9


def test_parse_and_match_ratings__rejects_non_letterboxd_csv() -> None:
    with pytest.raises(LetterboxdExportError, match="does not look"):
        parse_and_match_ratings(b"title,score\nA,5\n", [], str(uuid4()), "ratings.csv")
