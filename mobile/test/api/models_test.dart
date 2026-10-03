import 'package:flutter_test/flutter_test.dart';
import 'package:movie_recommender_mobile/api/models.dart';

Map<String, dynamic> _movieJson({
  String? posterUrl = 'http://example.com/p.jpg',
  List<dynamic> genres = const ['Action', 'Drama'],
  List<dynamic> actors = const ['Actor A'],
}) {
  return {
    'id': 1,
    'title': 'Some Movie',
    'release_year': 2024,
    'genres': genres,
    'director': 'Director A',
    'country': 'US',
    'actors': actors,
    'poster_url': posterUrl,
  };
}

void main() {
  test('test_movieFromJson__parsesFullMetadata', () {
    final movie = Movie.fromJson(_movieJson());

    expect(movie.id, 1);
    expect(movie.title, 'Some Movie');
    expect(movie.releaseYear, 2024);
    expect(movie.genres, ['Action', 'Drama']);
    expect(movie.director, 'Director A');
    expect(movie.country, 'US');
    expect(movie.actors, ['Actor A']);
    expect(movie.posterUrl, 'http://example.com/p.jpg');
  });

  test('test_movieFromJson__nullPosterUrl', () {
    final movie = Movie.fromJson(_movieJson(posterUrl: null));

    expect(movie.posterUrl, isNull);
  });

  test('test_movieFromJson__parsesEmptyGenresAndActors', () {
    final movie = Movie.fromJson(_movieJson(genres: const [], actors: const []));

    expect(movie.genres, isEmpty);
    expect(movie.actors, isEmpty);
  });

  test('test_recommendationFromJson__nullMatchScore', () {
    final recommendation = Recommendation.fromJson({
      'movie': _movieJson(),
      'reason': null,
      'match_score': null,
    });

    expect(recommendation.matchScore, isNull);
    expect(recommendation.reason, isNull);
    expect(recommendation.movie.title, 'Some Movie');
  });
}
