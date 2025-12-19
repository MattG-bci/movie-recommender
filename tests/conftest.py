import pytest

from schemas.modelling import ModelConfig


@pytest.fixture
def mock_model_config() -> ModelConfig:
    return ModelConfig(
        n_users=100,
        n_movies=100,
        embedding_dim=64,
        learning_rate=0.001,
    )
