-- Revert movie-recommender:11102025_add_genre_column_to_movies from pg

BEGIN;

-- XXX Add DDLs here.

ALTER TABLE movies DROP COLUMN genre;

DROP TRIGGER IF EXISTS updated_at_trig ON users;

DROP FUNCTION IF EXISTS updated_at_trig();

DROP TRIGGER IF EXISTS created_at_trig ON users;

DROP FUNCTION IF EXISTS created_at_trig();

COMMIT;
