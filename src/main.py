import typer

from etl.ingestion import ingest_movies, ingest_usernames, ingest_movie_ratings
import asyncio
from functools import wraps
import logging
import torch

from etl.sql_queries import (
    fetch_movie_ratings_from_db,
    fetch_movies_from_db,
    fetch_usernames_from_db,
)
from model.dataloader import construct_datasets
from model.train import train_movie_recommender, get_device, preprocess_movie_ratings
from model.recommender import (
    prepare_model_config,
    Recommender,
    get_model_id_to_recommender_id_mapping,
)
from schemas.modelling import TrainConfig, ModelConfig, PATH_TO_MODEL_WEIGHTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
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
    ratings = await fetch_movie_ratings_from_db()
    movies = await fetch_movies_from_db()
    user_names = await fetch_usernames_from_db()
    device = get_device()

    model_config = prepare_model_config(movies, user_names)
    model = Recommender(model_config)
    processed_ratings = preprocess_movie_ratings(ratings, movies, user_names)
    train_dataset, val_dataset = construct_datasets(processed_ratings)
    train_config = TrainConfig(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )
    model = train_movie_recommender(train_config)
    logging.info("Saving trained model...")
    torch.save(model.state_dict(), PATH_TO_MODEL_WEIGHTS)


@app.command()
@async_typer_command
async def recommend_movies(user_name: str, top_k: int = 5):
    movies = await fetch_movies_from_db()
    user_names = await fetch_usernames_from_db()

    map_movie_id_to_recommender_id = get_model_id_to_recommender_id_mapping(
        movies, "id"
    )
    map_user_id_to_recommender_id = get_model_id_to_recommender_id_mapping(
        user_names, "id"
    )
    map_user_name_to_id = {
        user.username: map_user_id_to_recommender_id[user.id] for user in user_names
    }
    map_movie_id_to_name = {
        map_movie_id_to_recommender_id[movie.id]: movie.title for movie in movies
    }

    movie_ids = list({map_movie_id_to_recommender_id[movie.id] for movie in movies})
    n_movies = len(movie_ids)
    n_users = len({user.id for user in user_names})

    user_id = map_user_name_to_id.get(user_name)
    if user_id is None:
        raise KeyError(f"User name {user_name} does not exist in the database")

    model_config = ModelConfig(n_users=n_users, n_movies=n_movies)
    state_dict = torch.load(PATH_TO_MODEL_WEIGHTS, map_location=torch.device("cpu"))
    model = Recommender(model_config)
    model.load_state_dict(state_dict)
    user_id = torch.tensor(user_id).to(torch.device("cpu"))
    movie_ids = torch.tensor(movie_ids).to(torch.device("cpu"))
    recommendations = model.get_top_k_recommendations(user_id, movie_ids, top_k)

    movie_names = [
        map_movie_id_to_name.get(int(recommended_movie_id))
        for recommended_movie_id in recommendations
    ]
    logging.info(f"Here is top-{top_k} recommended movies: {movie_names}")


if __name__ == "__main__":
    app()
