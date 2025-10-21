-- Deploy movie-recommender:21102025_add_constraints to pg

BEGIN;

-- XXX Add DDLs here.

ALTER TABLE movies ADD CONSTRAINT unique_movie UNIQUE (title, release_year, director);

ALTER TABLE movie_ratings ADD CONSTRAINT unique_rating UNIQUE (user_id, movie_id);

COMMIT;
