# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git conventions

Do not append a `Claude-Session:` trailer (or any session URL) to commit messages or PR descriptions in this repo.

## File structure

```
movie-recommender/
├── .circleci/
├── .github/
│   └── workflows/
├── dags/
├── models/
├── sqitch/
├── src/
│   ├── api/
│   │   └── routers/
│   ├── etl/
│   │   └── generation/
│   ├── model/
│   ├── schemas/
│   └── utils/
├── tests/
│   ├── fixtures/
│   ├── test_api/
│   │   └── test_routers/
│   ├── test_etl/
│   ├── test_model/
│   └── test_scraper/
├── .env_template
├── .gitattributes
├── .pre-commit-config.yaml
├── CLAUDE.md
├── Dockerfile
├── README.md
├── justfile
├── poetry.lock
└── pyproject.toml
```

## Project overview

A movie recommender system: scrapes user/movie/rating data, stores it in Postgres, trains a
collaborative-filtering (CF) model in PyTorch, and reranks CF candidates with an LLM (via DSPy + Anthropic) based
on a free-text prompt and optional image. Exposes a small FastAPI service and a Typer CLI for the ETL/training/
recommendation pipeline.

## Commands

Dependency management is via Poetry (`pyproject.toml` + `poetry.lock`); `uv` is used to run the API.

```bash
# Tests (spins up docker-compose services: postgres, fake nginx scraping site, API container)
just test                    # pytest -n 4 tests/
poetry run pytest tests/test_model/test_train.py::test_name   # single test

# Lint / format (ruff; also runs in pre-commit and CI)
just lint                    # poetry run ruff format src/ tests/

# Database migrations (sqitch, config in sqitch/sqitch.conf)
just deploy-db
just revert-db
just verify-db

# Run the ETL/training/recommendation CLI (Typer app in src/main.py)
just pipeline run_usernames_ingestion
just pipeline run_movies_ingestion
just pipeline run_ratings_ingestion
just pipeline run_all_ingestion
just pipeline run_recommender_training
just pipeline run_movie_recommendation <username> <img_path> [--exploration 0.3] [--prompt "..."]

# Run the API locally
just api                     # uv run uvicorn api.app:app --reload
```

Tests require Docker (`tests/docker-compose.yml` brings up `test-db`, `test-api`, and a `fake-site` nginx
container serving fixture HTML for scraper tests). `tests/conftest.py` handles xdist-safe setup/teardown via a
file lock so containers are shared across the 4 parallel workers, deploys the schema with sqitch, and loads CSV
fixtures from `tests/fixtures/db/` (UUIDs are deterministically derived via `uuid.uuid5` to match the app's UUID
primary keys).

## Architecture

**`src/settings.py`** — Pydantic `BaseSettings` classes (`DBSettings`, `WebScraperSettings`, `LLMSettings`), each
with its own env prefix (`DB_`, `SCRAPER_`, `ANTHROPIC_`), loaded from `.env` at the repo root.

**ETL / ingestion (`src/etl/`)**
- `generation/web_scraping.py` — Playwright + BeautifulSoup scrapers for users, movies, and ratings
  (with `tenacity` retries and `playwright-stealth`).
- `generation/generate.py` — orchestrates scraping + dedup against what's already in the DB.
- `sql_queries.py` — `DatabaseConnector` (async/sync context manager wrapping psycopg2/asyncpg) plus
  upsert/fetch helpers for movies, users, and ratings.
- `ingestion.py` — top-level `ingest_movies` / `ingest_usernames` / `ingest_movie_ratings`, called from the CLI.

**Modelling (`src/model/`)**
- `recommender.py` — `CFRecommender` (PyTorch embedding-dot-product model with bias terms) for candidate
  generation, and `MovieReranker` (DSPy module) which chains an image-semantics extraction step into an
  LLM reranking step.
- `train.py` / `dataloader.py` / `processing.py` — training loop and data prep (mapping DB ids <-> dense
  recommender ids used by the embedding tables).
- `llm_rerank.py` — `recommend_movies`: full pipeline — fetch data → CF candidates → build `MovieCandidate`s →
  rerank with the LLM → merge/fill remaining slots so exactly `top_k_recommendations` are returned.
- `evaluate.py` — model evaluation.

**Schemas (`src/schemas/`)** — Pydantic models/dataclasses shared across layers: `Movie`/`MovieIn`, `User`/`UserIn`,
`ModelConfig` + `PATH_TO_MODEL_WEIGHTS` (`modelling.py`), and DSPy `Signature`s for reranking (`RerankMovies`,
`UserProfile`, `MovieCandidate`, `ExtractImageSemantics`) in `recommendation.py`. Model weights are checked into
`models/recommender_model.pth`.

**API (`src/api/`)** — FastAPI app (`app.py`) with routers under `api/routers/` (`healthcheck.py`, `ratings.py`).

**CLI (`src/main.py`)** — Typer app; each command is an async function wrapped with `async_typer_command` to run
via `asyncio.run`. This is the single entry point tying ETL, training, and recommendation together.

**Database migrations (`sqitch/`)** — sqitch-managed SQL migrations (`deploy/`, `revert/`, `verify/` triplets),
tracked in `sqitch.plan`. IDs are UUIDs (see the `add_uuid_columns` migration); fixture loading in tests derives
matching UUIDs with `uuid.uuid5` so foreign keys line up.

**Airflow (`dags/`)** — `ingestion_dag.py` schedules the ETL ingestion pipeline.
