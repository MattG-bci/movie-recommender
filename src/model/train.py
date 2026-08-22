from collections import defaultdict

import torch
import torch.nn as nn

from model.dataloader import construct_datasets
from model.evaluate import calculate_metrics
from model.recommender import (
    logger,
    get_model_id_to_recommender_id_mapping,
    prepare_model_config,
    CFRecommender,
)
from schemas.modelling import TrainConfig, TrainHyperparameters
from schemas.movie import MovieRatingWithId, Movie
from schemas.users import User
from utils.model_size import timeit


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def preprocess_movie_ratings(
    ratings: list[MovieRatingWithId], movies: list[Movie], users: list[User]
) -> list[MovieRatingWithId]:
    map_movie_id_to_recommender_id = get_model_id_to_recommender_id_mapping(
        movies, "id"
    )
    map_user_id_to_recommender_id = get_model_id_to_recommender_id_mapping(users, "id")
    ratings = [
        rating.model_copy(
            update={
                "user_id": map_user_id_to_recommender_id[rating.user_id],
                "movie_id": map_movie_id_to_recommender_id[rating.movie_id],
            }
        )
        for rating in ratings
    ]
    return ratings


@timeit
def train_movie_recommender(config: TrainConfig) -> nn.Module:
    model = config.model
    device = config.device
    model.to(device)
    optimizer = model.optimiser
    criterion = model.loss

    for epoch in range(config.hyperparams.epochs):
        logger.info(f"--------EPOCH {epoch + 1}/{config.hyperparams.epochs}--------")
        train_metrics = defaultdict(list)
        model.train()
        for batch_idx, (batch_user_ids, batch_movie_ids, batch_ratings) in enumerate(
            config.train_dataloader
        ):
            batch_user_ids = batch_user_ids.to(device)
            batch_movie_ids = batch_movie_ids.to(device)
            batch_ratings = batch_ratings.to(device)

            train_preds = model(batch_user_ids, batch_movie_ids)
            loss = criterion(train_preds, batch_ratings)

            train_metrics["predictions"].extend(train_preds.detach().cpu().numpy())
            train_metrics["targets"].extend(batch_ratings.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info("----Training Metrics----")
        metrics = calculate_metrics(
            train_metrics["predictions"], train_metrics["targets"]
        )
        for metric in metrics.model_fields.keys():
            logger.info(f"Train {metric.upper()}: {getattr(metrics, metric):.3f}")

        logger.info("Starting validation...")
        validation_metrics = defaultdict(list)
        model.eval()
        for batch_idx, (batch_user_ids, batch_movie_ids, batch_ratings) in enumerate(
            config.val_dataloader
        ):
            batch_user_ids = batch_user_ids.to(device)
            batch_movie_ids = batch_movie_ids.to(device)
            batch_ratings = batch_ratings.to(device)

            val_preds = model(batch_user_ids, batch_movie_ids)

            validation_metrics["predictions"].extend(val_preds.detach().cpu().numpy())
            validation_metrics["targets"].extend(batch_ratings.detach().cpu().numpy())

        logger.info("----Validation Metrics----")
        metrics = calculate_metrics(
            validation_metrics["predictions"], validation_metrics["targets"]
        )
        for metric in metrics.model_fields.keys():
            logger.info(f"Validation {metric.upper()}: {getattr(metrics, metric):.3f}")
        logger.info("--------------------------")
    logger.info("Training complete.")
    return model


def prepare_train_config_for_cfrecommender(
    user_names: list[User], movies: list[Movie], ratings: list[MovieRatingWithId]
) -> TrainConfig:
    hyperparams = TrainHyperparameters()

    logger.info("Constructing dataloaders for recommender training..")
    processed_ratings = preprocess_movie_ratings(ratings, movies, user_names)
    train_dataset, val_dataset = construct_datasets(processed_ratings)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=hyperparams.batch_size, shuffle=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=hyperparams.batch_size, shuffle=False
    )

    logger.info("Preparing model...")
    model_config = prepare_model_config(movies, user_names)
    model = CFRecommender(model_config)

    device = get_device()
    train_config = TrainConfig(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        hyperparams=hyperparams,
    )
    return train_config
