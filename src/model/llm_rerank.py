import dspy

from etl.sql_queries import fetch_user_profile
from model.recommender import MovieReranker
from settings import LLMSettings
from schemas.recommendation import RecommendationOut, MovieCandidate


async def rerank(
    user_id: int,
    prompt: str,
    exploration: float,
    candidates: list[MovieCandidate],
    k: int = 10,
) -> list[RecommendationOut]:
    user_profile = await fetch_user_profile(user_id, top_k=k)
    reranker = MovieReranker()

    candidate_payload = [
        {
            "movie_id": c.movie.id,
            "title": c.movie.title,
            "genres": c.movie.genres,
            "director": c.movie.director,
            "actors": c.movie.actors,
            "cf_score": c.cf_score,
        }
        for c in candidates
    ]
    by_id = {c.movie.id: c for c in candidates}

    configure_llm()
    prediction = reranker(
        request=prompt,
        exploration=exploration,
        user_profile=user_profile.model_dump(),
        candidates=candidate_payload,
    )

    results: list[RecommendationOut] = []
    seen: set[int] = set()
    for mid in prediction.ranked_ids:
        if mid in by_id and mid not in seen:
            results.append(
                RecommendationOut(
                    movie=by_id[mid].movie, reason=prediction.reasons.get(mid)
                )
            )
            seen.add(mid)

    for c in candidates:
        if len(results) >= k:
            break
        if c.movie.id not in seen:
            results.append(RecommendationOut(movie=c.movie, reason=None))
            seen.add(c.movie.id)
    return results[:k]


def configure_llm() -> None:
    settings = LLMSettings()
    lm = dspy.LM(
        f"anthropic/{settings.MODEL}", api_key=settings.API_KEY, max_tokens=1024
    )
    dspy.configure(lm=lm)
