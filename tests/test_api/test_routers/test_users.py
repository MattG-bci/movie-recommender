import requests


def test_get_user__existing_username(api_url):
    out = requests.get(f"{api_url}/users/testuser1")
    body = out.json()
    assert out.status_code == 200
    assert body["username"] == "testuser1"
    assert "id" in body


def test_get_user__unknown_username_returns_404(api_url):
    out = requests.get(f"{api_url}/users/no_such_user")
    assert out.status_code == 404
