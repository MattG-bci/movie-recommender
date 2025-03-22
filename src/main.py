import typer
from src.etl.ingestion.ingest_usernames import ingest_usernames
import asyncio


app = typer.Typer(no_args_is_help=True)


@app.command()
def ingest_usrs() -> None:
    asyncio.run(ingest_usernames())


@app.command()
def run_all_ingestion() -> None:
    asyncio.run(ingest_usernames())


if __name__ == "__main__":
    app()
