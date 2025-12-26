import logging
from collections import defaultdict

import torch
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EvalMetrics(BaseModel):
    mse: float
    mape: float


def calculate_metrics(
    predictions: list[torch.Tensor], targets: list[torch.Tensor]
) -> EvalMetrics:
    calculated_metrics = defaultdict(float)
    for metric in EvalMetrics.model_fields.keys():
        match metric:
            case "mse":
                mse = calculate_mse(predictions, targets)
                calculated_metrics["mse"] = mse
            case "mape":
                mape = calculate_mape(predictions, targets)
                calculated_metrics["mape"] = mape
            case _:
                raise KeyError("The requested metric has no implementation yet.")
    return EvalMetrics(**calculated_metrics)


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
