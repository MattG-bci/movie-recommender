from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from api.routers import health_router, ratings_router

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
