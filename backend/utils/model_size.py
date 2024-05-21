import torch
import pytorch_lightning as pl


def compute_model_size(model: pl.LightningModule) -> float:
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = num_params * 4 / (1024**2)
    return round(model_size_mb, 2)

