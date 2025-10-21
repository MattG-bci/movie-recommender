-- Revert movie-recommender:21102025_add_constraints from pg

BEGIN;

-- XXX Add DDLs here.

ALTER TABLE movies DROP CONSTRAINT IF EXISTS unique_movie;

ALTER TABLE movie_ratings DROP CONSTRAINT IF EXISTS unique_rating;

COMMIT;
