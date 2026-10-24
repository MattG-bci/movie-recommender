-- Verify movie-recommender:24102026_promote_uuid_ids on pg

BEGIN;

SELECT id FROM users WHERE FALSE;
SELECT id FROM movies WHERE FALSE;
SELECT id, user_id, movie_id FROM movie_ratings WHERE FALSE;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name IN ('users', 'movies', 'movie_ratings')
          AND column_name = 'id_uuid'
    ) THEN
        RAISE EXCEPTION 'legacy id_uuid columns remain';
    END IF;
END;
$$;

ROLLBACK;
