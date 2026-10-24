-- Verify movie-recommender:19102026_add_movie_recommendation_embeddings on pg

BEGIN;

SELECT movie_id, recommendation_profile, embedding_model, embedding
FROM movie_embeddings
WHERE FALSE;

ROLLBACK;
