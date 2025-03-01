from src.utils.model_size import compute_model_size
from model.recommender import Recommender


def test_compute_model_size():
    recommender = Recommender(100, 100)
    assert compute_model_size(recommender) == 0.05
