import os

import pytest


@pytest.fixture
def fixtures_path() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "fixtures")
