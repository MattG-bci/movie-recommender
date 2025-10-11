-- Deploy movie-recommender:11102025_add_genre_column_to_movies to pg

BEGIN;

-- XXX Add DDLs here.

ALTER TABLE movies ADD COLUMN genre VARCHAR(100);

CREATE FUNCTION updated_at_trig() RETURNS trigger
   LANGUAGE plpgsql AS
$$BEGIN
   NEW.updated_at := current_timestamp;
   RETURN NEW;
END;$$;


CREATE FUNCTION created_at_trig() RETURNS trigger
   LANGUAGE plpgsql AS
$$BEGIN
   NEW.created_at := current_timestamp;
   RETURN NEW;
END;$$;

CREATE TRIGGER created_at_trig BEFORE INSERT ON users
   FOR EACH ROW EXECUTE PROCEDURE created_at_trig();

CREATE TRIGGER updated_at_trig BEFORE UPDATE ON users
   FOR EACH ROW EXECUTE PROCEDURE updated_at_trig();

CREATE TRIGGER created_at_trig BEFORE INSERT ON movies
   FOR EACH ROW EXECUTE PROCEDURE created_at_trig();

CREATE TRIGGER updated_at_trig BEFORE UPDATE ON movies
   FOR EACH ROW EXECUTE PROCEDURE updated_at_trig();

CREATE TRIGGER created_at_trig BEFORE INSERT ON movie_ratings
    FOR EACH ROW EXECUTE PROCEDURE created_at_trig();

CREATE TRIGGER updated_at_trig BEFORE UPDATE ON movie_ratings
    FOR EACH ROW EXECUTE PROCEDURE updated_at_trig();

COMMIT;
