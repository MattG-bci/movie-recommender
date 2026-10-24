from functools import lru_cache

import dspy

from etl.sql_queries import (
    DatabaseConnector,
    fetch_movie_embeddings,
    fetch_movies_from_db,
    fetch_usernames_from_db,
    upsert_movie_embeddings,
)
from model.processing import map_recommender_movie_ids_to_db_ids
from model.recommender import (
    ImageSemanticExtractor,
    get_cf_recommendations,
    load_cf_recommender,
)
from model.semantic import (
    EmbeddingClient,
    blend_scores,
    build_movie_profile,
    score_semantic_query,
)
from schemas.movie import Movie
from schemas.recommendation import (
    MovieEmbedding,
    RecommendationInput,
    RecommendationOut,
)
from schemas.users import User
from settings import DBSettings, EmbeddingSettings, LLMSettings


async def recommend_movies(
    recommendation_input: RecommendationInput,
) -> list[RecommendationOut]:
    movies, users, stored = await load_data_from_db_for_recommendation(DBSettings())
    model = load_cf_recommender(len(users), len(movies))
    recommender_ids = list(range(len(movies)))
    target_user_id = get_target_user_id(users, recommendation_input.username)
    predicted_ids, values = get_cf_recommendations(
        model, target_user_id, recommender_ids, len(movies)
    )
    movie_ids = map_recommender_movie_ids_to_db_ids(movies, predicted_ids)
    cf_scores = dict(zip(movie_ids, (float(value) for value in values)))

    text_query = recommendation_input.prompt.strip()
    image_query = extract_image_query(recommendation_input.image)
    if not text_query and not image_query:
        return format_recommendations(
            movies, cf_scores, recommendation_input.top_k_recommendations
        )

    movie_vectors = {
        item.movie_id: item.embedding
        for item in validate_embeddings(stored, EmbeddingSettings().MODEL)
    }
    if set(movie_vectors) != set(cf_scores):
        raise ValueError(
            "Movie embeddings are missing or use a different model. "
            "Run the offline movie embedding backfill first."
        )
    semantic_queries = {
        name: query
        for name, query in {"text": text_query, "image": image_query}.items()
        if query
    }
    query_embeddings = EmbeddingClient().embed(list(semantic_queries.values()))
    semantic_scores = {
        name: score_semantic_query(embedding, movie_vectors)
        for name, embedding in zip(semantic_queries.keys(), query_embeddings)
    }

    scores = blend_scores(
        cf_scores,
        text_scores=semantic_scores.get("text"),
        image_scores=semantic_scores.get("image"),
    )
    return format_recommendations(
        movies, scores, recommendation_input.top_k_recommendations
    )


async def load_data_from_db_for_recommendation(
    settings: DBSettings,
) -> tuple[list[Movie], list[User], list[MovieEmbedding]]:
    async with DatabaseConnector(db_settings=settings) as conn:
        movies = await fetch_movies_from_db(conn)
        users = await fetch_usernames_from_db(conn)
        embeddings = await fetch_movie_embeddings(conn)
    return movies, users, embeddings


def get_target_user_id(users: list[User], username: str) -> int:
    for index, user in enumerate(sorted(users, key=lambda item: item.id)):
        if user.username == username:
            return index
    raise ValueError(f"Unknown username: {username}")


def validate_embeddings(
    embeddings: list[MovieEmbedding], embedding_model: str
) -> list[MovieEmbedding]:
    if any(
        item.recommendation_profile is None
        or item.embedding is None
        or item.embedding_model != embedding_model
        for item in embeddings
    ):
        raise ValueError(
            "Movie embeddings are missing or use a different model. "
            "Run the offline movie embedding backfill first."
        )
    return embeddings


def extract_image_query(image: dspy.Image | None) -> str | None:
    if image is None:
        return None
    with dspy.context(lm=get_lm()):
        semantics = ImageSemanticExtractor()(image=image)
    parts = [
        semantics.description,
        f"Mood: {semantics.mood}",
        f"Energy: {semantics.energy}",
        f"Suggested genres: {', '.join(semantics.suggested_genres)}",
    ]
    if semantics.avoid_genres:
        parts.append(f"Avoid: {', '.join(semantics.avoid_genres)}")
    return ". ".join(parts)


def format_recommendations(
    movies: list[Movie], scores: dict[int, float], k: int
) -> list[RecommendationOut]:
    by_id = {movie.id: movie for movie in movies}
    movie_ids = sorted(scores, key=lambda movie_id: (-scores[movie_id], movie_id))
    return [
        RecommendationOut(
            movie=by_id[movie_id], reason=None, match_score=scores[movie_id]
        )
        for movie_id in movie_ids[:k]
    ]


async def backfill_movie_embeddings() -> None:
    settings = DBSettings()
    async with DatabaseConnector(db_settings=settings) as conn:
        movies = await fetch_movies_from_db(conn)
        profiles = [build_movie_profile(movie) for movie in movies]
        vectors = EmbeddingClient().embed(profiles)
        embedding_settings = EmbeddingSettings()
        embeddings = [
            MovieEmbedding(
                movie_id=movie.id,
                recommendation_profile=profile,
                embedding_model=embedding_settings.MODEL,
                embedding=vector,
            )
            for movie, profile, vector in zip(movies, profiles, vectors)
        ]
        await upsert_movie_embeddings(conn, embeddings)


@lru_cache(maxsize=1)
def get_lm() -> dspy.LM:
    settings = LLMSettings()
    return dspy.LM(f"anthropic/{settings.MODEL}", api_key=settings.API_KEY)
