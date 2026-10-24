-- Deploy movie-recommender:19102026_add_movie_recommendation_embeddings to pg

BEGIN;

ALTER TABLE movies
    ADD COLUMN description TEXT,
    ADD CONSTRAINT unique_movie_id_uuid UNIQUE (id_uuid);

CREATE TABLE movie_embeddings (
    movie_id UUID PRIMARY KEY REFERENCES movies(id_uuid) ON DELETE CASCADE,
    recommendation_profile TEXT NOT NULL,
    embedding_model VARCHAR(255) NOT NULL,
    embedding DOUBLE PRECISION[] NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

COMMIT;
