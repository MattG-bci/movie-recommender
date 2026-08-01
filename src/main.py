from logging import getLogger

import typer

from etl.ingestion import ingest_movies, ingest_usernames, ingest_movie_ratings
import asyncio
from functools import wraps
import logging
import torch

from etl.sql_queries import (
    DatabaseConnector,
    fetch_movie_ratings_from_db,
    fetch_movies_from_db,
    fetch_usernames_from_db,
)
from model.dataloader import construct_datasets
from model.llm_rerank import rerank
from model.train import train_movie_recommender, get_device, preprocess_movie_ratings
from model.recommender import (
    prepare_model_config,
    CFRecommender,
    get_model_id_to_recommender_id_mapping,
)
from schemas.modelling import TrainConfig, ModelConfig, PATH_TO_MODEL_WEIGHTS
from schemas.recommendation import MovieCandidate
from settings import DBSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = getLogger(__name__)

app = typer.Typer(no_args_is_help=True)


def async_typer_command(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


@app.command()
@async_typer_command
async def ingest_users() -> None:
    await ingest_usernames()


@app.command()
@async_typer_command
async def ingest_movies_command() -> None:
    await ingest_movies()


@app.command()
@async_typer_command
async def ingest_ratings() -> None:
    await ingest_movie_ratings()


@app.command()
@async_typer_command
async def run_all_ingestion() -> None:
    await ingest_usernames()
    await ingest_movies()
    await ingest_movie_ratings()


@app.command()
@async_typer_command
async def train_recommender() -> None:
    async with DatabaseConnector() as conn:
        ratings = await fetch_movie_ratings_from_db(conn)
        movies = await fetch_movies_from_db(conn)
        user_names = await fetch_usernames_from_db(conn)

    device = get_device()
    model_config = prepare_model_config(movies, user_names)
    model = CFRecommender(model_config)
    processed_ratings = preprocess_movie_ratings(ratings, movies, user_names)
    train_dataset, val_dataset = construct_datasets(processed_ratings)
    train_config = TrainConfig(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )
    model = train_movie_recommender(train_config)
    logger.info("Saving trained model...")
    torch.save(model.state_dict(), PATH_TO_MODEL_WEIGHTS)


@app.command()
@async_typer_command
async def recommend_movies(
    user_name: str, top_k: int = 10, exploration: float = 1.0, n_cf_candidates: int = 40
):
    settings = DBSettings()
    async with DatabaseConnector(db_settings=settings) as conn:
        movies = await fetch_movies_from_db(conn)
        user_names = await fetch_usernames_from_db(conn)

    map_movie_id_to_recommender_id = get_model_id_to_recommender_id_mapping(
        movies, "id"
    )
    map_user_id_to_recommender_id = get_model_id_to_recommender_id_mapping(
        user_names, "id"
    )
    map_recommender_id_to_movie_id = {
        value: key for key, value in map_movie_id_to_recommender_id.items()
    }

    map_user_name_to_db_id = {user.username: user.id for user in user_names}

    movie_ids = list({map_movie_id_to_recommender_id[movie.id] for movie in movies})
    n_movies = len(movie_ids)
    n_users = len({user.id for user in user_names})

    database_user_id = map_user_name_to_db_id.get(user_name)
    if database_user_id is None:
        raise KeyError(f"User name {user_name} does not exist in the database")
    recommender_user_id = map_user_id_to_recommender_id[database_user_id]

    model_config = ModelConfig(n_users=n_users, n_movies=n_movies)
    state_dict = torch.load(PATH_TO_MODEL_WEIGHTS, map_location=torch.device("cpu"))
    model = CFRecommender(model_config)
    model.load_state_dict(state_dict)
    user_id_tensor = torch.tensor([recommender_user_id]).to(torch.device("cpu"))
    movie_ids = torch.tensor(movie_ids).to(torch.device("cpu"))
    recommendations, cf_scores = model.get_top_k_recommendations(
        user_id_tensor, movie_ids, k=n_cf_candidates
    )

    recommended_movie_ids = [
        map_recommender_id_to_movie_id.get(int(recommended_movie_id))
        for recommended_movie_id in recommendations
    ]

    logger.info("Reranking...")
    candidates: list[MovieCandidate] = []
    for recommended_movie_id, score in zip(recommended_movie_ids, cf_scores):
        movie = list(filter(lambda x: x.id == recommended_movie_id, movies))
        if not movie:
            continue
        candidate = MovieCandidate(movie=movie[0], cf_score=float(score.detach()))
        candidates.append(candidate)

    prompt = "I want something relaxing"
    reranked_recommendations = await rerank(
        database_user_id, prompt, exploration, candidates=candidates, k=top_k
    )
    logger.info(
        f"Here is top {top_k} movie recommendations after reranking: {reranked_recommendations}"
    )


if __name__ == "__main__":
    app()
