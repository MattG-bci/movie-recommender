# Movie Recommender — mobile

Flutter client for the movie recommender. Two tabs once signed in: **Explore**
(browse the catalogue with posters and metadata) and **Recommend** (prompt +
optional photo + exploration slider -> ranked results with a match score).

## Requirements

- Flutter >= 3.22, Dart >= 3.4
- iOS and Android targets only (no web/desktop)
- Bundle id: `com.movierecommender.mobile`

## Running

```bash
cd mobile
flutter pub get
flutter run --dart-define=API_BASE_URL=http://localhost:8080
```

`API_BASE_URL` defaults to `http://localhost:8080`. A physical device needs
the host machine's LAN IP instead of `localhost`, and both platforms need a
cleartext-HTTP exemption unless the API is fronted by TLS (the API has no TLS
of its own).

## Testing

`just test` at the repo root only runs the **Python** suite; it does not
cover this directory. Run the Dart suite separately:

```bash
just test-mobile
# or, equivalently:
cd mobile && flutter test
```

## Posters

Posters are resolved server-side from OMDb (see `src/api/poster_service.py`)
and degrade to `null` — rendered here as a designed fallback tile — when the
server has no `OMDB_API_KEY` configured, OMDb has no match, or the daily
quota is exhausted. There is no direct OMDb access from the app.

**Docker caveat**: the server's poster cache (`OMDB_CACHE_PATH`, default
`poster_cache.json` next to the API process) must be pointed at a mounted
volume when running the API in a container, or every container rebuild
throws the cache away and restarts the OMDb quota spend from zero.
