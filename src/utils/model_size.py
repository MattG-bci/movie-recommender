from torch import nn
import time


def compute_model_size(model: nn.Module) -> float:
    """This method outputs the size of a model in MBs."""
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = num_params * 4 / (1024**2)
    return round(model_size_mb, 2)


def timeit(func):
    def inner(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        return result

    return inner
