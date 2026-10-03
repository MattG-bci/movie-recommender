import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:movie_recommender_mobile/api/api_client.dart';
import 'package:movie_recommender_mobile/api/models.dart';
import 'package:movie_recommender_mobile/state/movies_controller.dart';

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

  test('test_loadFirstPage__populatesMovies', () async {
    when(() => api.fetchMovies(limit: any(named: 'limit'), offset: 0, search: null))
        .thenAnswer(
      (_) async => MoviePage(
        items: [_makeMovie(1), _makeMovie(2)],
        total: 2,
        limit: 30,
        offset: 0,
      ),
    );
    final controller = MoviesController(api: api);

    await controller.loadFirstPage();

    expect(controller.movies.length, 2);
    expect(controller.isLoading, isFalse);
    expect(controller.error, isNull);
  });

  test('test_loadNextPage__appendsAndAdvancesOffset', () async {
    when(() => api.fetchMovies(limit: any(named: 'limit'), offset: 0, search: null))
        .thenAnswer(
      (_) async => MoviePage(items: [_makeMovie(1)], total: 2, limit: 1, offset: 0),
    );
    when(() => api.fetchMovies(limit: any(named: 'limit'), offset: 1, search: null))
        .thenAnswer(
      (_) async => MoviePage(items: [_makeMovie(2)], total: 2, limit: 1, offset: 1),
    );
    final controller = MoviesController(api: api, pageSize: 1);
    await controller.loadFirstPage();

    await controller.loadNextPage();

    expect(controller.movies.map((m) => m.id), [1, 2]);
    verify(() => api.fetchMovies(limit: any(named: 'limit'), offset: 1, search: null))
        .called(1);
  });

  test('test_loadNextPage__noOpWhenNoMorePages', () async {
    when(() => api.fetchMovies(limit: any(named: 'limit'), offset: 0, search: null))
        .thenAnswer(
      (_) async => MoviePage(items: [_makeMovie(1)], total: 1, limit: 30, offset: 0),
    );
    final controller = MoviesController(api: api);
    await controller.loadFirstPage();

    await controller.loadNextPage();

    verifyNever(() => api.fetchMovies(limit: any(named: 'limit'), offset: 1, search: any(named: 'search')));
  });

  test('test_loadNextPage__noOpWhileLoading', () async {
    when(() => api.fetchMovies(limit: any(named: 'limit'), offset: 0, search: null))
        .thenAnswer((_) async {
      await Future<void>.delayed(const Duration(milliseconds: 50));
      return MoviePage(items: [_makeMovie(1)], total: 5, limit: 1, offset: 0);
    });
    final controller = MoviesController(api: api, pageSize: 1);

    final firstLoad = controller.loadFirstPage();
    final secondCallDuringLoad = controller.loadNextPage();
    await Future.wait([firstLoad, secondCallDuringLoad]);

    verifyNever(() => api.fetchMovies(limit: any(named: 'limit'), offset: 1, search: any(named: 'search')));
  });

  test('test_setSearch__debouncesAndResetsOffset', () async {
    when(() => api.fetchMovies(limit: any(named: 'limit'), offset: 0, search: 'matrix'))
        .thenAnswer(
      (_) async => MoviePage(items: [_makeMovie(1)], total: 1, limit: 30, offset: 0),
    );
    final controller = MoviesController(api: api);

    await controller.setSearch('m');
    await controller.setSearch('ma');
    await controller.setSearch('matrix');
    await Future<void>.delayed(const Duration(milliseconds: 400));

    verify(() => api.fetchMovies(limit: any(named: 'limit'), offset: 0, search: 'matrix'))
        .called(1);
  });

  test('test_loadFirstPage__exposesErrorOnApiException', () async {
    when(() => api.fetchMovies(limit: any(named: 'limit'), offset: 0, search: null))
        .thenThrow(ApiException(500, 'boom'));
    final controller = MoviesController(api: api);

    await controller.loadFirstPage();

    expect(controller.error, isA<ApiException>());
    expect(controller.isLoading, isFalse);
  });
}
