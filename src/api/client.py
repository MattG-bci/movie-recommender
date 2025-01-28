import httpx


class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url)

    def get(self, path: str):
        return self.client.get(path)

    def post(self, path: str, data: dict):
        return self.client.post(path, json=data)

    def put(self, path: str, data: dict):
        return self.client.put(path, json=data)

    def delete(self, path: str):
        return self.client.delete(path)
