from utils.model_size import compute_model_size
from model.recommender import Recommender


def test_compute_model_size(mock_model_config):
    recommender = Recommender(mock_model_config)
    assert compute_model_size(recommender) == 0.05
