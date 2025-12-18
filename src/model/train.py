import torch
from torch import optim
from torch.utils.data import DataLoader

from model.dataloader import construct_datasets_for_train_eval
from model.recommender import Recommender, logger
from schemas.modelling import ConfigTrain
from schemas.movie import MovieRating
from utils.model_size import timeit


def prepare_data(ratings: list[MovieRating]) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = construct_datasets_for_train_eval(ratings)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False
    )
    return train_dataloader, val_dataloader


def prepare_model(ratings: list[MovieRating]) -> Recommender:
    n_users = len({rating.user_id for rating in ratings})
    n_movies = len({rating.movie_id for rating in ratings})
    model = Recommender(n_users, n_movies)
    return model


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


@timeit
def train_movie_recommender(config: ConfigTrain, epochs: int = 50) -> None:
    train_dataloader = config.train_dataloader
    val_dataloader = config.val_dataloader
    model = config.model
    device = config.device

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = model.loss

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        train_loss = []
        for batch_idx, (batch_user_ids, batch_movie_ids, batch_ratings) in enumerate(
            train_dataloader
        ):
            batch_user_ids = batch_user_ids.to(device)
            batch_movie_ids = batch_movie_ids.to(device)
            batch_ratings = batch_ratings.to(device)

            train_preds = model(batch_user_ids, batch_movie_ids)
            loss = criterion(train_preds, batch_ratings)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_train_loss = sum(train_loss) / len(train_loss)
        logger.info(f"Training loss: {mean_train_loss:.3f}")

        validation_loss = []
        for batch_idx, (batch_user_ids, batch_movie_ids, batch_ratings) in enumerate(
            val_dataloader
        ):
            batch_user_ids = batch_user_ids.to(device)
            batch_movie_ids = batch_movie_ids.to(device)
            batch_ratings = batch_ratings.to(device)

            val_preds = model(batch_user_ids, batch_movie_ids)
            val_loss = criterion(val_preds, batch_ratings)
            validation_loss.append(val_loss.item())

        mean_val_loss = sum(validation_loss) / len(validation_loss)
        logger.info(f"Validation loss: {mean_val_loss:.3f}")
