-- Revert movie-recommender:19102026_add_movie_recommendation_embeddings from pg

BEGIN;

DROP TABLE movie_embeddings;

ALTER TABLE movies
    DROP CONSTRAINT unique_movie_id_uuid,
    DROP COLUMN description;

COMMIT;
