import logging

import torch
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EvalMetrics(BaseModel):
    loss: float
    mse: float


def calculate_metrics(metrics: dict[str, list[torch.tensor]]) -> EvalMetrics:
    mean_loss = sum(metrics["loss"]) / len(metrics["loss"])
    preds = metrics["predictions"]
    targets = metrics["targets"]
    mse = calculate_mse(preds, targets)
    return EvalMetrics(loss=mean_loss, mse=mse)


def calculate_mse(preds: list[torch.tensor], targets: list[torch.tensor]) -> float:
    preds_tensor = torch.tensor(preds).view(-1)
    targets_tensor = torch.tensor(targets).view(-1)
    mse = torch.mean((preds_tensor - targets_tensor) ** 2).item()
    return mse
