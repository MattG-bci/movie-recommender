import dspy
import typer

from etl.ingestion import ingest_movies, ingest_usernames, ingest_movie_ratings
import asyncio
from functools import wraps
import logging
import torch

from model.train import train_recommender
from model.llm_rerank import recommend_movies

from schemas.modelling import PATH_TO_MODEL_WEIGHTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True)


def async_typer_command(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


@app.command()
@async_typer_command
async def run_usernames_ingestion() -> None:
    await ingest_usernames()


@app.command()
@async_typer_command
async def run_movies_ingestion() -> None:
    await ingest_movies()


@app.command()
@async_typer_command
async def run_ratings_ingestion() -> None:
    await ingest_movie_ratings()


@app.command()
@async_typer_command
async def run_all_ingestion() -> None:
    await ingest_usernames()
    await ingest_movies()
    await ingest_movie_ratings()


@app.command()
@async_typer_command
async def run_recommender_training(save_model: bool = True) -> None:
    model = await train_recommender()
    if save_model:
        logger.info("Saving trained model...")
        torch.save(model.state_dict(), PATH_TO_MODEL_WEIGHTS)


@app.command()
@async_typer_command
async def run_movie_recommendation(
    user_name: str,
    img_path: str | None,
    top_k: int = 10,
    exploration: float = 0.3,
    n_cf_candidates: int = 100,
    prompt: str = "I am a bit lonely now. I need something light-hearted",
) -> None:
    img = dspy.Image(img_path) if img_path else None
    await recommend_movies(
        user_name=user_name,
        top_k=top_k,
        exploration=exploration,
        n_cf_candidates=n_cf_candidates,
        prompt=prompt,
        image=img,
    )


if __name__ == "__main__":
    app()
