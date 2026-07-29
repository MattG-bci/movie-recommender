import pytest

from etl.generation.web_scraping import MovieScraper


@pytest.mark.asyncio
async def test_scrape_movies(fake_site_url):
    scraper = MovieScraper(movie_page_url=f"{fake_site_url}/films/popular/")

    movies = await scraper.get_data_incremental(existing_movie_titles=[])

    assert movies is not None
    assert len(movies) == 3

    titles = [m.title for m in movies]
    assert "The Matrix" in titles
    assert "Inception" in titles
    assert "Pulp Fiction" in titles


@pytest.mark.asyncio
async def test_movie_metadata(fake_site_url):
    scraper = MovieScraper(movie_page_url=f"{fake_site_url}/films/popular/")

    movies = await scraper.get_data_incremental(existing_movie_titles=[])
    movie_map = {m.title: m for m in movies}

    matrix = movie_map["The Matrix"]
    assert matrix.release_year == 1999
    assert matrix.director == "Lana Wachowski"
    assert matrix.country == "USA"
    assert "Action" in matrix.genres
    assert "Science Fiction" in matrix.genres
    assert "Keanu Reeves" in matrix.actors
    assert len(matrix.actors) <= 5

    inception = movie_map["Inception"]
    assert inception.release_year == 2010
    assert inception.director == "Christopher Nolan"


@pytest.mark.asyncio
async def test_skips_existing_movies(fake_site_url):
    scraper = MovieScraper(movie_page_url=f"{fake_site_url}/films/popular/")

    movies = await scraper.get_data_incremental(existing_movie_titles=["The Matrix"])

    assert movies is not None
    titles = [m.title for m in movies]
    assert "The Matrix" not in titles
    assert "Inception" in titles
    assert "Pulp Fiction" in titles
