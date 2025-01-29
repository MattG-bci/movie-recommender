-- Revert movie-recommender:29012025_create_initial_schema from pg

BEGIN;

-- XXX Add DDLs here.

DROP TABLE IF EXISTS movie_ratings CASCADE;
DROP TABLE IF EXISTS movies CASCADE;
DROP TABLE IF EXISTS users CASCADE;

COMMIT;
