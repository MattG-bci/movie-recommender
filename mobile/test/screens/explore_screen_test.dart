import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:movie_recommender_mobile/api/api_client.dart';
import 'package:movie_recommender_mobile/api/models.dart';
import 'package:movie_recommender_mobile/screens/explore_screen.dart';

class _MockApiClient extends Mock implements ApiClient {}

Movie _makeMovie(int id, {String title = 'Movie'}) {
  return Movie(
    id: id,
    title: '$title $id',
    releaseYear: 2020,
    genres: const ['Action'],
    director: 'Director',
    country: 'US',
    actors: const ['Actor'],
  );
}

void main() {
  late _MockApiClient api;

  setUp(() {
    api = _MockApiClient();
  });

  testWidgets('test_exploreScreen__rendersMovieTitles', (tester) async {
    when(() => api.fetchMovies(
          limit: any(named: 'limit'),
          offset: any(named: 'offset'),
          search: any(named: 'search'),
        )).thenAnswer(
      (_) async => MoviePage(
        items: [_makeMovie(1, title: 'The Matrix'), _makeMovie(2, title: 'Inception')],
        total: 2,
        limit: 30,
        offset: 0,
      ),
    );

    await tester.pumpWidget(MaterialApp(home: ExploreScreen(api: api)));
    await tester.pumpAndSettle();

    expect(find.textContaining('The Matrix'), findsOneWidget);
    expect(find.textContaining('Inception'), findsOneWidget);
  });

  testWidgets('test_exploreScreen__showsEmptyStateWhenSearchHasNoResults', (tester) async {
    when(() => api.fetchMovies(
          limit: any(named: 'limit'),
          offset: any(named: 'offset'),
          search: null,
        )).thenAnswer(
      (_) async => MoviePage(items: [_makeMovie(1)], total: 1, limit: 30, offset: 0),
    );
    when(() => api.fetchMovies(
          limit: any(named: 'limit'),
          offset: any(named: 'offset'),
          search: 'no-such-movie',
        )).thenAnswer(
      (_) async => const MoviePage(items: [], total: 0, limit: 30, offset: 0),
    );

    await tester.pumpWidget(MaterialApp(home: ExploreScreen(api: api)));
    await tester.pumpAndSettle();

    await tester.enterText(find.byType(TextField), 'no-such-movie');
    await tester.pump(const Duration(milliseconds: 400));
    await tester.pumpAndSettle();

    expect(find.text('No movies found'), findsOneWidget);
  });

  testWidgets('test_exploreScreen__showsRetryOnError', (tester) async {
    when(() => api.fetchMovies(
          limit: any(named: 'limit'),
          offset: any(named: 'offset'),
          search: any(named: 'search'),
        )).thenThrow(ApiException(500, 'boom'));

    await tester.pumpWidget(MaterialApp(home: ExploreScreen(api: api)));
    await tester.pumpAndSettle();

    expect(find.text('Retry'), findsOneWidget);
  });
}
