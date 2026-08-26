import csv
import subprocess
from pathlib import Path


from etl.sql_queries import DatabaseConnector
from schemas.modelling import ModelConfig

import pytest
import requests
import asyncio
import psycopg2

from pytest_docker.plugin import DockerComposeExecutor, Services
from filelock import FileLock

from settings import DBSettings


SQITCH_PATH = Path(__file__).parents[1] / "sqitch"
FIXTURES_PATH = Path(__file__).parent / "fixtures" / "db"


@pytest.fixture(scope="session")
def docker_compose_file():
    return str(Path(__file__).parent / "docker-compose.yml")


@pytest.fixture(scope="session")
def docker_compose_command():
    return "docker-compose"


@pytest.fixture(scope="session")
def docker_compose_project_name():
    return "test-movie-recommender"


@pytest.fixture(scope="session")
def docker_services(
    docker_compose_command,
    docker_compose_file,
    docker_compose_project_name,
    tmp_path_factory,
):
    docker_compose = DockerComposeExecutor(
        docker_compose_command, docker_compose_file, docker_compose_project_name
    )
    # All xdist workers share the same tmp root; use a lock so only the
    # first worker tears down old containers and brings up new ones.
    root_tmp = tmp_path_factory.getbasetemp().parent
    lock_file = root_tmp / "docker.lock"
    flag_file = root_tmp / "docker.flag"

    with FileLock(str(lock_file)):
        if not flag_file.exists():
            docker_compose.execute("down -v")
            docker_compose.execute("up --build -d")
            flag_file.write_text("started")
    yield Services(docker_compose)


def is_site_responsive(url: str) -> bool:
    try:
        resp = requests.get(url, timeout=2)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


@pytest.fixture(scope="session")
def fake_site_url(docker_ip, docker_services):
    port = docker_services.port_for("fake-site", 80)
    base_url = f"http://{docker_ip}:{port}"

    docker_services.wait_until_responsive(
        timeout=15,
        pause=0.5,
        check=lambda: is_site_responsive(
            f"{base_url}/members/popular/this/week/page/1/"
        ),
    )
    return base_url


@pytest.fixture(autouse=True)
def scraper_env(fake_site_url, monkeypatch):
    """Override scraper env vars to point at the fake nginx site."""
    monkeypatch.setenv("SCRAPER_BASE_URL", fake_site_url)
    monkeypatch.setenv(
        "SCRAPER_USERNAME_PAGE",
        f"{fake_site_url}/members/popular/this/week/",
    )
    monkeypatch.setenv("SCRAPER_RATINGS_PAGE", f"{fake_site_url}/")
    monkeypatch.setenv("SCRAPER_MOVIES_PAGE", f"{fake_site_url}/films/popular/")


@pytest.fixture(autouse=True)
def db_env(monkeypatch):
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_USER", "postgres")
    monkeypatch.setenv("DB_PASS", "postgres")
    monkeypatch.setenv("DB_NAME", "test")
    monkeypatch.setenv("DB_PORT", "54320")


async def create_db(settings: DBSettings):
    async with DatabaseConnector(db_settings=settings) as conn:
        await conn.execute("""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = 'test' AND pid <> pg_backend_pid()
        """)
        await conn.execute("DROP DATABASE IF EXISTS test;")

    async with DatabaseConnector(db_settings=settings) as conn:
        await conn.execute("CREATE DATABASE test;")

    new_settings = settings.model_copy()
    new_settings.NAME = "test"
    return new_settings


@pytest.fixture(scope="session")
def db_service(docker_ip, docker_services, tmp_path_factory) -> DBSettings:
    port = docker_services.port_for("test-db", 5432)

    docker_services.wait_until_responsive(
        timeout=30, pause=1, check=lambda: is_db_responsive(docker_ip, port)
    )
    settings = DBSettings(
        USER="postgres",
        PASS="postgres",
        NAME="postgres",
        HOST=docker_ip,
        PORT=port,
    )

    root_tmp = tmp_path_factory.getbasetemp().parent
    lock_file = root_tmp / "db_setup.lock"
    flag_file = root_tmp / "db_setup.flag"

    with FileLock(str(lock_file)):
        if not flag_file.exists():
            new_settings = asyncio.run(create_db(settings))
            setup_test_db(new_settings)
            load_fixtures(new_settings)
            flag_file.write_text("done")
        else:
            new_settings = settings.model_copy()
            new_settings.NAME = "test"

    return new_settings


def is_db_responsive(host, port) -> bool:
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user="postgres",
            password="postgres",
            database="postgres",
            connect_timeout=2,
        )
        conn.close()
        return True
    except Exception as e:
        print(f"DB not ready: {e}")
        return False


@pytest.fixture
def mock_model_config() -> ModelConfig:
    return ModelConfig(
        n_users=100,
        n_movies=100,
        embedding_dim=64,
        learning_rate=0.001,
    )


def run_sqitch(settings: DBSettings, command: str):
    dsn = settings.get_postgres_dsn("db:pg")
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "-v",
            f"{SQITCH_PATH}:/repo",
            "sqitch/sqitch:latest",
            command,
            "--target",
            dsn,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Sqitch {command} failed for the test database: {result.returncode}\n"
            f"{result.stdout} \n----\n {result.stderr}"
        )


def setup_test_db(settings: DBSettings):
    run_sqitch(settings, "deploy")


def load_fixtures(settings: DBSettings):
    # Order of the tables matters here
    fixtures = [
        ("users", ["id", "username"]),
        (
            "movies",
            ["id", "title", "release_year", "director", "genres", "country", "actors"],
        ),
        ("movie_ratings", ["id", "user_id", "movie_id", "rating"]),
    ]

    for table, columns in fixtures:
        with open(FIXTURES_PATH / f"{table}.csv") as f:
            data = list(csv.DictReader(f))

        values = ", ".join(f"%({col})s" for col in columns)
        cols = ", ".join(columns)
        query = f"INSERT INTO {table} ({cols}) VALUES ({values})"

        with DatabaseConnector(db_settings=settings) as conn:
            conn.cursor().executemany(query, data)
            conn.commit()


@pytest.fixture(scope="session")
def api_url(docker_ip, docker_services, db_service):
    port = docker_services.port_for("test-api", 8080)
    url = f"http://{docker_ip}:{port}"
    docker_services.wait_until_responsive(
        timeout=30,
        pause=1,
        check=lambda: is_site_responsive(f"{url}/health"),
    )
    return url
