import asyncpg
import pytest
import asyncio
import contextlib
import psycopg2
from settings import DBSettings

from pytest_docker.plugin import DockerComposeExecutor, Services


@pytest.fixture(scope="session")
def docker_compose_file():
    return "tests/docker-compose.yml"


@pytest.fixture(scope="session")
def docker_compose_command():
    return "docker-compose"


@pytest.fixture(scope="session")
def docker_compose_project_name():
    return "test-db"


@contextlib.contextmanager
def get_docker_services(
    docker_compose_command,
    docker_compose_file,
    docker_compose_project_name,
):
    docker_compose = DockerComposeExecutor(
        docker_compose_command, docker_compose_file, docker_compose_project_name
    )
    try:
        yield Services(docker_compose)
    finally:
        pass


@pytest.fixture(scope="session")
def docker_services(
    docker_compose_command,
    docker_compose_file,
    docker_compose_project_name,
):
    with get_docker_services(
        docker_compose_command,
        docker_compose_file,
        docker_compose_project_name,
    ) as services:
        yield services


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
        print(f"DB not ready: {e}")  # Helpful for debugging
        return False


async def create_db(settings: DBSettings):
    conn = await asyncpg.connect(settings.get_postgres_dsn("postgresql"))
    assert conn is not None
    await conn.execute("DROP DATABASE IF EXISTS test;")
    await conn.execute("CREATE DATABASE test;")
    await conn.close()
    new_settings = settings.model_copy()
    new_settings.NAME = "test"
    return new_settings


@pytest.mark.asyncio
async def test_db_connection(db_service):
    conn = await asyncpg.connection.connect(
        user=db_service.USER,
        password=db_service.PASS,
        database=db_service.NAME,
        host=db_service.HOST,
        port=db_service.PORT,
        timeout=5,
    )
    assert conn is not None
    await conn.close()
