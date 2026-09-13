import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:mocktail/mocktail.dart';
import 'package:movie_recommender_mobile/api/api_client.dart';

class _MockHttpClient extends Mock implements http.Client {}

void main() {
  late _MockHttpClient httpClient;
  late ApiClient apiClient;

  setUp(() {
    httpClient = _MockHttpClient();
    apiClient = ApiClient(client: httpClient, baseUrl: 'http://test-host');
  });

  setUpAll(() {
    registerFallbackValue(Uri.parse('http://test-host'));
  });

  test('test_fetchMovies__buildsQueryParameters', () async {
    when(() => httpClient.get(any())).thenAnswer(
      (_) async => http.Response(
        jsonEncode({'items': [], 'total': 0, 'limit': 20, 'offset': 5}),
        200,
      ),
    );

    await apiClient.fetchMovies(limit: 20, offset: 5, search: 'matrix');

    final captured = verify(() => httpClient.get(captureAny())).captured;
    final uri = captured.single as Uri;
    expect(uri.path, '/movies');
    expect(uri.queryParameters['limit'], '20');
    expect(uri.queryParameters['offset'], '5');
    expect(uri.queryParameters['search'], 'matrix');
  });

  test('test_fetchMovies__throwsApiExceptionOnServerError', () async {
    when(() => httpClient.get(any())).thenAnswer(
      (_) async => http.Response('boom', 500),
    );

    expect(
      () => apiClient.fetchMovies(),
      throwsA(isA<ApiException>()),
    );
  });

  test('test_fetchUser__throwsNotFoundExceptionOn404', () async {
    when(() => httpClient.get(any())).thenAnswer(
      (_) async => http.Response('not found', 404),
    );

    expect(
      () => apiClient.fetchUser('unknown'),
      throwsA(isA<NotFoundException>()),
    );
  });

  test('test_recommend__omitsImageWhenNull', () async {
    when(() => httpClient.post(any(), headers: any(named: 'headers'), body: any(named: 'body')))
        .thenAnswer((_) async => http.Response('[]', 200));

    await apiClient.recommend(username: 'alice', prompt: 'something', exploration: 0.3);

    final captured = verify(
      () => httpClient.post(any(), headers: any(named: 'headers'), body: captureAny(named: 'body')),
    ).captured;
    final body = jsonDecode(captured.single as String) as Map<String, dynamic>;
    expect(body.containsKey('image_base64'), isFalse);
  });

  test('test_recommend__postsBase64ImageWhenProvided', () async {
    when(() => httpClient.post(any(), headers: any(named: 'headers'), body: any(named: 'body')))
        .thenAnswer((_) async => http.Response('[]', 200));

    await apiClient.recommend(
      username: 'alice',
      prompt: 'something',
      exploration: 0.3,
      imageBase64: 'ZmFrZQ==',
    );

    final captured = verify(
      () => httpClient.post(any(), headers: any(named: 'headers'), body: captureAny(named: 'body')),
    ).captured;
    final body = jsonDecode(captured.single as String) as Map<String, dynamic>;
    expect(body['image_base64'], 'ZmFrZQ==');
  });
}
