import torch

from model.evaluate import calculate_mape
from src.model.evaluate import calculate_metrics, calculate_mse


def test_calculate_metrics():
    mock_metrics = {
        "predictions": [torch.tensor(3.0), torch.tensor(4.0), torch.tensor(5.0)],
        "targets": [torch.tensor(2.0), torch.tensor(5.0), torch.tensor(4.0)],
    }
    result = calculate_metrics(mock_metrics["predictions"], mock_metrics["targets"])
    expected_mse = 1.0  # manually calculated
    assert result.mse == expected_mse


def test_calculate_mse():
    mock_targets = [
        torch.tensor(3.0),
        torch.tensor(5.0),
        torch.tensor(2.0),
        torch.tensor(4.0),
    ]
    mock_preds = [
        torch.tensor(2.5),
        torch.tensor(4.5),
        torch.tensor(2.0),
        torch.tensor(5.0),
    ]
    result = calculate_mse(mock_preds, mock_targets)

    expected = 0.375  # manually calculated
    assert result == expected


def test_calculate_mape():
    mock_targets = [
        torch.tensor(3.0),
        torch.tensor(5.0),
        torch.tensor(2.0),
        torch.tensor(4.0),
    ]
    mock_preds = [
        torch.tensor(2.5),
        torch.tensor(4.5),
        torch.tensor(2.0),
        torch.tensor(5.0),
    ]
    result = calculate_mape(mock_preds, mock_targets)

    expected = 0.129  # manually calculated
    assert round(result, 3) == expected
