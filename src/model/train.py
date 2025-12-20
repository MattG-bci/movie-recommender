from collections import defaultdict

import torch
import torch.nn as nn

from model.evaluate import calculate_metrics
from model.recommender import logger
from schemas.modelling import TrainConfig
from utils.model_size import timeit


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


@timeit
def train_movie_recommender(config: TrainConfig) -> nn.Module:
    train_dataloader = torch.utils.data.DataLoader(
        config.train_dataset, batch_size=config.batch_size, shuffle=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        config.val_dataset, batch_size=config.batch_size, shuffle=False
    )
    model = config.model
    device = config.device

    model.to(device)
    optimizer = model.optimiser
    criterion = model.loss

    for epoch in range(config.epochs):
        logger.info(f"--------EPOCH {epoch + 1}/{config.epochs}--------")
        train_metrics = defaultdict(list)
        for batch_idx, (batch_user_ids, batch_movie_ids, batch_ratings) in enumerate(
            train_dataloader
        ):
            batch_user_ids = batch_user_ids.to(device)
            batch_movie_ids = batch_movie_ids.to(device)
            batch_ratings = batch_ratings.to(device)

            train_preds = model(batch_user_ids, batch_movie_ids)
            loss = criterion(train_preds, batch_ratings)

            train_metrics["loss"].append(loss.item())
            train_metrics["predictions"].extend(train_preds.detach().cpu().numpy())
            train_metrics["targets"].extend(batch_ratings.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info("----Training Metrics----")
        metrics = calculate_metrics(train_metrics)
        logger.info(f"Train Loss: {metrics.loss:.3f}")
        logger.info(f"Train MSE: {metrics.mse:.3f}")

        logger.info("Starting validation...")
        validation_metrics = defaultdict(list)
        for batch_idx, (batch_user_ids, batch_movie_ids, batch_ratings) in enumerate(
            val_dataloader
        ):
            batch_user_ids = batch_user_ids.to(device)
            batch_movie_ids = batch_movie_ids.to(device)
            batch_ratings = batch_ratings.to(device)

            val_preds = model(batch_user_ids, batch_movie_ids)
            val_loss = criterion(val_preds, batch_ratings)

            validation_metrics["loss"].append(val_loss.item())
            validation_metrics["predictions"].extend(val_preds.detach().cpu().numpy())
            validation_metrics["targets"].extend(batch_ratings.detach().cpu().numpy())

        logger.info("----Validation Metrics----")
        metrics = calculate_metrics(validation_metrics)
        logger.info(f"Validation Loss: {metrics.loss:.3f}")
        logger.info(f"Validation MSE: {metrics.mse:.3f}")
        logger.info("--------------------------")
    logger.info("Training complete.")
    return model
