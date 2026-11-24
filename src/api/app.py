from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from api.routers import (
    health_router,
    ratings_router,
    movies_router,
    users_router,
    recommendations_router,
    imports_router,
)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # The separate browser frontend sends bearer tokens, rather than cookies.
    # Wildcard origins are valid in that mode and keep the generated Container
    # Apps hostname from needing to be known at API build time.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="")
app.include_router(ratings_router, prefix="/ratings")
app.include_router(movies_router, prefix="/movies")
app.include_router(users_router, prefix="/users")
app.include_router(recommendations_router, prefix="/recommendations")
app.include_router(imports_router, prefix="/imports")
