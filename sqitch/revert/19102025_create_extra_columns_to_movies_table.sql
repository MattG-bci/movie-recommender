-- Revert movie-recommender:19102025_create_extra_columns_to_movies_table from pg

BEGIN;

-- XXX Add DDLs here.
ALTER TABLE movies
    DROP COLUMN IF EXISTS director,
    DROP COLUMN IF EXISTS genres,
    DROP COLUMN IF EXISTS country,
    DROP COLUMN IF EXISTS actors;

ALTER TABLE movies ADD COLUMN genre VARCHAR(100);

COMMIT;
