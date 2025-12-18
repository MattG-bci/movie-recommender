import logging


logger = logging.getLogger(__name__)


def calculate_metrics(metrics: dict[str, list[float]]) -> None:
    mean_loss = sum(metrics["loss"]) / len(metrics["loss"])
    preds = metrics["predictions"]
    targets = metrics["targets"]
    mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(targets)

    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"Mean Loss: {mean_loss:.4f}")
