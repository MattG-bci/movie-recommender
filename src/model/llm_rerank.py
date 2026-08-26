import dspy

from etl.sql_queries import (
    fetch_user_profile,
    DatabaseConnector,
    fetch_movies_from_db,
    fetch_usernames_from_db,
)
from model.processing import prepare_recommendation_mappings, build_movie_candidates
from model.recommender import (
    MovieReranker,
    logger,
    load_cf_recommender,
    get_cf_recommendations,
)
from settings import LLMSettings, DBSettings
from schemas.recommendation import RecommendationOut, MovieCandidate


async def rerank(
    user_id: int,
    prompt: str,
    exploration: float,
    candidates: list[MovieCandidate],
    k: int = 10,
) -> list[RecommendationOut]:
    async with DatabaseConnector() as conn:
        user_profile = await fetch_user_profile(conn, user_id, top_k=k)

    reranker = MovieReranker()
    by_id = {c.movie.id: c for c in candidates}

    configure_llm()
    prediction = reranker(
        request=prompt,
        exploration=exploration,
        user_profile=user_profile,
        candidates=candidates,
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
    lm = dspy.LM(f"anthropic/{settings.MODEL}", api_key=settings.API_KEY)
    dspy.configure(lm=lm)


async def recommend_movies(
    user_name: str, top_k: int, exploration: float, n_cf_candidates: int, prompt: str
) -> None:
    settings = DBSettings()
    async with DatabaseConnector(db_settings=settings) as conn:
        movies = await fetch_movies_from_db(conn)
        user_names = await fetch_usernames_from_db(conn)

    logger.info("Prepare data...")
    (
        map_recommender_id_to_movie_id,
        recommender_user_id,
        database_user_id,
        movie_ids,
        n_users,
        n_movies,
    ) = prepare_recommendation_mappings(movies, user_names, user_name)

    logger.info("Loading base recommendation model...")
    model = load_cf_recommender(n_users, n_movies)

    logger.info("Getting base recommendations...")
    recommended_movie_ids, cf_scores = get_cf_recommendations(
        model,
        recommender_user_id,
        movie_ids,
        map_recommender_id_to_movie_id,
        n_cf_candidates,
    )

    candidates = build_movie_candidates(recommended_movie_ids, cf_scores, movies)

    logger.info("Reranking...")
    reranked_recommendations = await rerank(
        database_user_id, prompt, exploration, candidates=candidates, k=top_k
    )
    logger.info(
        f"Here is top {top_k} movie recommendations after reranking: {reranked_recommendations}"
    )
