def test_health(mock_client):
    out = mock_client.get("/health")
    assert out.json() == {"status": 200}


def test_readiness(mock_client):
    out = mock_client.get("/readiness")
    assert out.json() == {"status": 200}
