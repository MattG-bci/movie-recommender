from pydantic import BaseModel, Field
from uuid import UUID

import dspy
from schemas.movie import Movie, MovieOut


class RecommendationInput(BaseModel):
    username: str
    prompt: str
    image: dspy.Image | None = None
    top_k_recommendations: int = 10


class RecommendationOut(BaseModel):
    movie: Movie
    reason: str | None
    match_score: float | None = None


class RecommendationRequest(BaseModel):
    username: str
    prompt: str
    exploration: float = Field(0.3, ge=0.0, le=1.0)
    image_base64: str | None = None
    top_k: int = Field(10, ge=1, le=25)


class RecommendationItem(BaseModel):
    movie: MovieOut
    reason: str | None
    match_score: float | None


class MovieCandidate(BaseModel):
    movie: Movie
    cf_score: float


class MovieEmbedding(BaseModel):
    movie_id: UUID | int
    recommendation_profile: str | None
    embedding_model: str | None
    embedding: list[float] | None


class ImageSemantics(BaseModel):
    energy: str
    mood: str
    suggested_genres: list[str]
    avoid_genres: list[str]
    description: str


class ExtractImageSemantics(dspy.Signature):
    image: dspy.Image = dspy.InputField(
        desc="Image describing the mood, vibe and current state which is used for reranking movies to help find more suitable candidates"
    )

    output: ImageSemantics = dspy.OutputField(
        desc="A concise movie-search query inferred from the image, including mood, energy, setting or visual style, and suggested or avoided genres"
    )


class ExtractPromptIntent(dspy.Signature):
    """Translate a user's request into language that appears in movie metadata."""

    prompt: str = dspy.InputField(
        desc="The user's free-form request for a movie recommendation"
    )

    search_query: str = dspy.OutputField(
        desc=(
            "A concise movie-search query. State the desired mood, tone, themes, "
            "energy, genres, and any things to avoid. Preserve explicit constraints "
            "and do not mention the user's personal situation. For example, a request "
            "to be cheered up should become 'uplifting, funny, warm, optimistic, "
            "light-hearted comfort movie; avoid bleak, tragic, or depressing films'."
        )
    )
