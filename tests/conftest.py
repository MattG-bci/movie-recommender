from schemas.modelling import ModelConfig

import pytest
import requests
import asyncio
import asyncpg
import psycopg2

from pytest_docker.plugin import DockerComposeExecutor, Services
from filelock import FileLock

from settings import DBSettings


@pytest.fixture(scope="session")
def docker_compose_file():
    return "tests/docker-compose.yml"


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
            docker_compose.execute("-v down")
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

    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_USER", "postgres")
    monkeypatch.setenv("DB_PASS", "postgres")
    monkeypatch.setenv("DB_NAME", "test")
    monkeypatch.setenv("DB_PORT", "54320")


async def create_db(settings: DBSettings):
    conn = await asyncpg.connect(settings.get_postgres_dsn("postgresql"))
    assert conn is not None
    await conn.execute("DROP DATABASE IF EXISTS test;")
    await conn.execute("CREATE DATABASE test;")
    await conn.close()
    new_settings = settings.model_copy()
    new_settings.NAME = "test"
    return new_settings


@pytest.fixture(scope="session")
def db_service(docker_ip, docker_services):
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

    new_settings = asyncio.run(create_db(settings))
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
