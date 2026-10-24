-- Revert movie-recommender:24102026_promote_uuid_ids from pg

BEGIN;

ALTER TABLE movie_embeddings
    DROP CONSTRAINT movie_embeddings_movie_id_fkey;

ALTER TABLE movie_ratings
    ADD COLUMN legacy_user_id INTEGER,
    ADD COLUMN legacy_movie_id INTEGER;

ALTER TABLE users
    ADD COLUMN legacy_id SERIAL,
    ADD CONSTRAINT users_legacy_id_key UNIQUE (legacy_id);

ALTER TABLE movies
    ADD COLUMN legacy_id SERIAL,
    ADD CONSTRAINT movies_legacy_id_key UNIQUE (legacy_id);

UPDATE movie_ratings ratings
SET
    legacy_user_id = users.legacy_id,
    legacy_movie_id = movies.legacy_id
FROM users, movies
WHERE ratings.user_id = users.id
  AND ratings.movie_id = movies.id;

ALTER TABLE movie_ratings
    DROP CONSTRAINT movie_ratings_user_id_fkey,
    DROP CONSTRAINT movie_ratings_movie_id_fkey,
    DROP CONSTRAINT unique_rating,
    DROP CONSTRAINT movie_ratings_pkey;

ALTER TABLE movie_ratings RENAME COLUMN id TO id_uuid;
ALTER TABLE movie_ratings RENAME COLUMN user_id TO user_id_uuid;
ALTER TABLE movie_ratings RENAME COLUMN movie_id TO movie_id_uuid;
ALTER TABLE movie_ratings RENAME COLUMN legacy_user_id TO user_id;
ALTER TABLE movie_ratings RENAME COLUMN legacy_movie_id TO movie_id;

ALTER TABLE movie_ratings
    ADD COLUMN id SERIAL PRIMARY KEY,
    ADD CONSTRAINT movie_ratings_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(legacy_id),
    ADD CONSTRAINT movie_ratings_movie_id_fkey FOREIGN KEY (movie_id) REFERENCES movies(legacy_id),
    ADD CONSTRAINT unique_rating UNIQUE (user_id, movie_id);

ALTER TABLE users
    DROP CONSTRAINT users_pkey;
ALTER TABLE users RENAME COLUMN id TO id_uuid;
ALTER TABLE users RENAME COLUMN legacy_id TO id;
ALTER TABLE users
    ADD PRIMARY KEY (id);

ALTER TABLE movies
    DROP CONSTRAINT movies_pkey;
ALTER TABLE movies RENAME COLUMN id TO id_uuid;
ALTER TABLE movies RENAME COLUMN legacy_id TO id;
ALTER TABLE movies
    ADD PRIMARY KEY (id),
    ADD CONSTRAINT unique_movie_id_uuid UNIQUE (id_uuid);

ALTER TABLE movie_embeddings
    ADD CONSTRAINT movie_embeddings_movie_id_fkey
        FOREIGN KEY (movie_id) REFERENCES movies(id_uuid) ON DELETE CASCADE;

COMMIT;
