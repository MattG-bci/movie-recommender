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


def test_post_user__ingests_username_from_upstream(api_url, delete_user_and_ratings):
    username = "api-ingestion-user"
    try:
        out = requests.post(f"{api_url}/users/{username}")

        assert out.status_code == 200

        saved = requests.get(f"{api_url}/users/{username}")
        assert saved.status_code == 200
        assert saved.json()["username"] == username
    finally:
        delete_user_and_ratings(username)
