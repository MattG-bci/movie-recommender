-- Revert movie-recommender:03092026_add_uuid_columns from pg

BEGIN;

-- XXX Add DDLs here.
ALTER TABLE users
DROP COLUMN id_uuid;

ALTER TABLE movies
DROP COLUMN id_uuid;

ALTER TABLE movie_ratings
DROP COLUMN id_uuid;

COMMIT;
