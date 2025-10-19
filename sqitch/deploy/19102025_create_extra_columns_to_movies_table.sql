-- Deploy movie-recommender:19102025_create_extra_columns_to_movies_table to pg

BEGIN;

-- XXX Add DDLs here.
ALTER TABLE movies DROP COLUMN IF EXISTS genre;

ALTER TABLE movies
    ADD COLUMN director VARCHAR(255),
    ADD COLUMN genres VARCHAR(64)[],
    ADD COLUMN country VARCHAR(64),
    ADD COLUMN actors VARCHAR(255)[];

COMMIT;
