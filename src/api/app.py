from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from api.routers import (
    health_router,
    ratings_router,
    movies_router,
    users_router,
    recommendations_router,
)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="")
app.include_router(ratings_router, prefix="/ratings")
app.include_router(movies_router, prefix="/movies")
app.include_router(users_router, prefix="/users")
app.include_router(recommendations_router, prefix="/recommendations")
