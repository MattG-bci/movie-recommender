from fastapi import APIRouter


health = APIRouter()


@health.get("/health")
async def get_healthcheck() -> dict[str, int]:
    return {"status": 200}


@health.get("/readiness")
async def get_readiness() -> dict[str, int]:
    return {"status": 200}
