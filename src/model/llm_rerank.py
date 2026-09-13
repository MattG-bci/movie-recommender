from functools import lru_cache

import dspy

from etl.sql_queries import (
    fetch_user_profile,
    DatabaseConnector,
    fetch_movies_from_db,
    fetch_usernames_from_db,
)
from model.processing import (
    prepare_ids_for_recommendation,
    build_movie_candidates,
    map_recommender_movie_ids_to_db_ids,
)
from model.recommender import (
    MovieReranker,
    logger,
    load_cf_recommender,
    get_cf_recommendations,
)
from schemas.movie import Movie
from schemas.users import User
from settings import LLMSettings, DBSettings
from schemas.recommendation import (
    RecommendationOut,
    MovieCandidate,
    RecommendationInput,
    UserProfile,
)


async def recommend_movies(
    recommendation_input: RecommendationInput,
) -> list[RecommendationOut]:
    logger.info("Fetch data from db...")
    settings = DBSettings()
    movies, users, user_profile = await load_data_from_db_for_recommendation(
        recommendation_input.username, settings
    )

    logger.info("Prepare data...")
    (recommender_movie_ids, recommender_user_ids, target_user_id) = (
        prepare_ids_for_recommendation(movies, users, recommendation_input.username)
    )

    logger.info("Loading base recommendation model...")
    n_users = len(users)
    n_movies = len(movies)
    model = load_cf_recommender(n_users, n_movies)

    logger.info("Getting base cf recommendations...")
    recommended_movie_ids, cf_scores = get_cf_recommendations(
        model,
        target_user_id,
        recommender_movie_ids,
        recommendation_input.n_cf_recommendations,
    )

    recommended_movie_ids = map_recommender_movie_ids_to_db_ids(
        movies, recommended_movie_ids
    )
    candidates = build_movie_candidates(recommended_movie_ids, cf_scores, movies)

    logger.info("Reranking...")
    reranked_recommendations = await rerank_candidates(
        user_profile=user_profile,
        prompt=recommendation_input.prompt,
        exploration=recommendation_input.exploration,
        candidates=candidates,
        k=recommendation_input.top_k_recommendations,
        image=recommendation_input.image,
    )
    return reranked_recommendations


async def load_data_from_db_for_recommendation(
    username: str, settings: DBSettings
) -> tuple[list[Movie], list[User], UserProfile]:
    async with DatabaseConnector(db_settings=settings) as conn:
        movies = await fetch_movies_from_db(conn)

    async with DatabaseConnector(db_settings=settings) as conn:
        users = await fetch_usernames_from_db(conn)

    async with DatabaseConnector(db_settings=settings) as conn:
        user_profile = await fetch_user_profile(conn, username)
    return movies, users, user_profile


async def rerank_candidates(
    user_profile: UserProfile,
    prompt: str,
    exploration: float,
    candidates: list[MovieCandidate],
    image: dspy.Image | None = None,
    k: int = 10,
) -> list[RecommendationOut]:
    reranker = MovieReranker()
    by_id = {c.movie.id: c for c in candidates}

    # dspy.configure() pins the global LM to the asyncio task that first calls
    # it, so a per-request call blows up under uvicorn (each request is its own
    # task). dspy.context() sets the LM for this call only, via contextvars.
    with dspy.context(lm=get_lm()):
        prediction = reranker(
            request=prompt,
            exploration=exploration,
            user_profile=user_profile,
            candidates=candidates,
            image=image,
        )

    results: list[RecommendationOut] = []
    seen: set[int] = set()
    for movie_id in prediction.ranked_ids:
        if movie_id in by_id and movie_id not in seen:
            results.append(
                RecommendationOut(
                    movie=by_id[movie_id].movie,
                    reason=prediction.reasons.get(movie_id),
                    match_score=by_id[movie_id].cf_score,
                )
            )
            seen.add(movie_id)

    for c in candidates:
        if len(results) >= k:
            break
        if c.movie.id not in seen:
            results.append(
                RecommendationOut(movie=c.movie, reason=None, match_score=c.cf_score)
            )
            seen.add(c.movie.id)
    return results[:k]


@lru_cache(maxsize=1)
def get_lm() -> dspy.LM:
    """Build the LM once per process; dspy.LM is stateless and reusable."""
    settings = LLMSettings()
    return dspy.LM(f"anthropic/{settings.MODEL}", api_key=settings.API_KEY)
