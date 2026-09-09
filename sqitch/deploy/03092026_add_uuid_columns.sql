-- Deploy movie-recommender:03092026_add_uuid_columns to pg

BEGIN;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- XXX Add DDLs here.

ALTER TABLE users
ADD COLUMN id_uuid uuid DEFAULT NULL;

ALTER TABLE movies
ADD COLUMN id_uuid uuid DEFAULT NULL;

ALTER TABLE movie_ratings
ADD COLUMN id_uuid uuid DEFAULT NULL;


UPDATE users
SET id_uuid = uuid_generate_v5(uuid_ns_dns(), username);

UPDATE movies
SET id_uuid = uuid_generate_v5(uuid_ns_dns(), CONCAT(title, release_year, COALESCE(director, '')));

UPDATE movie_ratings mr
SET id_uuid = uuid_generate_v5(
  uuid_ns_dns(),
  CONCAT(u.id_uuid, m.id_uuid)
)
FROM users u, movies m
WHERE mr.user_id = u.id AND mr.movie_id = m.id;

ALTER TABLE users ALTER COLUMN id_uuid SET NOT NULL;
ALTER TABLE movies ALTER COLUMN id_uuid SET NOT NULL;
ALTER TABLE movie_ratings ALTER COLUMN id_uuid SET NOT NULL;

COMMIT;
