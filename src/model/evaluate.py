import logging
from collections import defaultdict
from typing import Literal

import torch

logger = logging.getLogger(__name__)


SupportedEvalMetrics = Literal["mse", "mape", "rmse"]


def calculate_metrics(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    metrics: list[SupportedEvalMetrics],
) -> dict[SupportedEvalMetrics, float]:
    calculated_metrics = defaultdict(float)
    for metric in metrics:
        match metric:
            case "mse":
                mse = calculate_mse(predictions, targets)
                calculated_metrics[metric] = mse
            case "mape":
                mape = calculate_mape(predictions, targets)
                calculated_metrics[metric] = mape
            case "rmse":
                rmse = calculate_rmse(predictions, targets)
                calculated_metrics[metric] = rmse
            case _:
                raise KeyError("The requested metric has no implementation yet.")
    return calculated_metrics


def calculate_mse(preds: list[torch.Tensor], targets: list[torch.Tensor]) -> float:
    preds_tensor = torch.tensor(preds).view(-1)
    targets_tensor = torch.tensor(targets).view(-1)
    mse = torch.mean((preds_tensor - targets_tensor) ** 2).item()
    return mse


def calculate_mape(preds: list[torch.Tensor], targets: list[torch.Tensor]) -> float:
    preds_tensor = torch.tensor(preds).view(-1)
    targets_tensor = torch.tensor(targets).view(-1)
    mape = torch.mean((targets_tensor - preds_tensor).abs() / targets_tensor).item()
    return mape


def calculate_rmse(preds: list[torch.Tensor], targets: list[torch.Tensor]) -> float:
    preds_tensor = torch.tensor(preds).view(-1)
    targets_tensor = torch.tensor(targets).view(-1)
    mse = torch.sqrt(torch.mean((preds_tensor - targets_tensor) ** 2)).item()
    return mse
