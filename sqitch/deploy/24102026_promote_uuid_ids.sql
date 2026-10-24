-- Deploy movie-recommender:24102026_promote_uuid_ids to pg

BEGIN;

-- Populate UUID foreign keys before removing the legacy integer identifiers.
ALTER TABLE movie_ratings
    ADD COLUMN user_id_uuid UUID,
    ADD COLUMN movie_id_uuid UUID;

UPDATE movie_ratings ratings
SET
    user_id_uuid = users.id_uuid,
    movie_id_uuid = movies.id_uuid
FROM users, movies
WHERE ratings.user_id = users.id
  AND ratings.movie_id = movies.id;

ALTER TABLE movie_embeddings
    DROP CONSTRAINT movie_embeddings_movie_id_fkey;

ALTER TABLE movie_ratings
    DROP CONSTRAINT movie_ratings_user_id_fkey,
    DROP CONSTRAINT movie_ratings_movie_id_fkey,
    DROP CONSTRAINT unique_rating,
    DROP CONSTRAINT movie_ratings_pkey,
    DROP COLUMN id,
    DROP COLUMN user_id,
    DROP COLUMN movie_id;

ALTER TABLE movie_ratings RENAME COLUMN id_uuid TO id;
ALTER TABLE movie_ratings RENAME COLUMN user_id_uuid TO user_id;
ALTER TABLE movie_ratings RENAME COLUMN movie_id_uuid TO movie_id;

ALTER TABLE movie_ratings
    ADD PRIMARY KEY (id),
    ADD CONSTRAINT unique_rating UNIQUE (user_id, movie_id);

ALTER TABLE users
    DROP CONSTRAINT users_pkey,
    DROP COLUMN id;
ALTER TABLE users RENAME COLUMN id_uuid TO id;
ALTER TABLE users
    ADD PRIMARY KEY (id);

ALTER TABLE movies
    DROP CONSTRAINT unique_movie_id_uuid,
    DROP CONSTRAINT movies_pkey,
    DROP COLUMN id;
ALTER TABLE movies RENAME COLUMN id_uuid TO id;
ALTER TABLE movies
    ADD PRIMARY KEY (id);

ALTER TABLE movie_ratings
    ADD CONSTRAINT movie_ratings_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id),
    ADD CONSTRAINT movie_ratings_movie_id_fkey
        FOREIGN KEY (movie_id) REFERENCES movies(id);

ALTER TABLE movie_embeddings
    ADD CONSTRAINT movie_embeddings_movie_id_fkey
        FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE;

COMMIT;
