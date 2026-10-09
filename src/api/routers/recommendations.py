import base64
import binascii

import dspy
from fastapi import APIRouter, HTTPException

from api.poster_service import PosterBudget, PosterCache, resolve_posters
from model.llm_rerank import recommend_movies
from schemas.movie import MovieOut
from schemas.recommendation import (
    RecommendationInput,
    RecommendationItem,
    RecommendationRequest,
)
from settings import OMDbSettings

recommendations = APIRouter()

_omdb_settings = OMDbSettings()
_poster_cache = PosterCache(_omdb_settings.CACHE_PATH)
_poster_budget = PosterBudget(_omdb_settings.DAILY_BUDGET)


def decode_image(image_base64: str | None) -> dspy.Image | None:
    """Decode a bare base64 string or a `data:image/...;base64,...` data URL
    into a dspy.Image. Returns None untouched since prompt-only requests are
    the common path."""
    if image_base64 is None:
        return None

    payload = image_base64
    mime = "image/jpeg"
    if payload.startswith("data:") and ";base64," in payload:
        prefix, payload = payload.split(";base64,", 1)
        mime = prefix.removeprefix("data:") or mime

    # Strip whitespace/line breaks a client may have sent (e.g. MIME-wrapped
    # base64) so validate=True only rejects genuinely malformed data.
    payload = "".join(payload.split())

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Malformed base64 image") from exc

    normalised = base64.b64encode(raw).decode()
    return dspy.Image(f"data:{mime};base64,{normalised}")


@recommendations.post("")
async def post_recommendations(
    payload: RecommendationRequest,
) -> list[RecommendationItem]:
    image = decode_image(payload.image_base64)
    recommendation_input = RecommendationInput(
        username=payload.username,
        prompt=payload.prompt,
        image=image,
        exploration=payload.exploration,
        top_k_recommendations=payload.top_k,
    )

    try:
        results = await recommend_movies(recommendation_input)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    poster_urls = await resolve_posters(
        [r.movie for r in results], _omdb_settings, _poster_cache, _poster_budget
    )

    return [
        RecommendationItem(
            movie=MovieOut(
                **r.movie.model_dump(), poster_url=poster_urls.get(r.movie.id)
            ),
            reason=r.reason,
            match_score=r.match_score,
        )
        for r in results
    ]
