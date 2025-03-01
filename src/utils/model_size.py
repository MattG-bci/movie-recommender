from torch import nn

from src.model.recommender import Recommender


def compute_model_size(model: nn.Module) -> float:
    """This method outputs the size of a model in MBs."""
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = num_params * 4 / (1024**2)
    return round(model_size_mb, 2)


if __name__ == "__main__":
    recommender = Recommender(100, 100)
    print(compute_model_size(recommender))
