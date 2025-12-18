import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EvalMetrics(BaseModel):
    loss: float
    mse: float


def calculate_metrics(metrics: dict[str, list[float]]) -> EvalMetrics:
    mean_loss = sum(metrics["loss"]) / len(metrics["loss"])
    preds = metrics["predictions"]
    targets = metrics["targets"]
    mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(targets)
    return EvalMetrics(loss=mean_loss, mse=mse)
