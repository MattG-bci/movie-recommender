import requests


def test_health(api_url):
    out = requests.get(f"{api_url}/health")
    assert out.json() == {"status": 200}


def test_readiness(api_url):
    out = requests.get(f"{api_url}/readiness")
    assert out.json() == {"status": 200}
