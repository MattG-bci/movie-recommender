import torch

from src.model.evaluate import calculate_metrics, calculate_mse


def test_calculate_metrics():
    mock_metrics = {
        "loss": [torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.4)],
        "predictions": [torch.tensor(3.0), torch.tensor(4.0), torch.tensor(5.0)],
        "targets": [torch.tensor(2.0), torch.tensor(5.0), torch.tensor(4.0)],
    }
    result = calculate_metrics(mock_metrics)
    expected_loss = 0.4  # manually calculated
    expected_mse = 1.0  # manually calculated
    assert round(result.loss, 2) == expected_loss
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
