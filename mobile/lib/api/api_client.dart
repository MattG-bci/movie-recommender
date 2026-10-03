import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config.dart';
import 'models.dart';

/// Thrown for any non-2xx response.
class ApiException implements Exception {
  ApiException(this.statusCode, this.message);

  final int statusCode;
  final String message;

  @override
  String toString() => 'ApiException($statusCode): $message';
}

/// Thrown specifically for 404s so callers (e.g. the identify screen) can
/// tell "no such user" apart from "server down".
class NotFoundException extends ApiException {
  NotFoundException(String message) : super(404, message);
}

/// Thrown when a request outlives its timeout. Separate from [ApiException]
/// because the server may still be working - the recommendation round-trip
/// routinely takes ~50s and is slower still on the first call of a process,
/// when the CF model weights load and no posters are cached yet.
class ApiTimeoutException implements Exception {
  ApiTimeoutException(this.limit);

  final Duration limit;

  @override
  String toString() => 'ApiTimeoutException(after ${limit.inSeconds}s)';
}

/// Maps an error from [ApiClient] onto a message worth showing a user.
/// Anything unrecognised falls back to its `toString()` rather than being
/// flattened into "something went wrong", which hides the cause.
String describeApiError(Object error) {
  if (error is ApiTimeoutException) {
    return 'The server took longer than ${error.limit.inSeconds}s to respond. '
        'It may still be working - try again.';
  }
  if (error is NotFoundException) {
    return 'Not found.';
  }
  if (error is ApiException) {
    return 'Server error ${error.statusCode}: ${error.message}';
  }
  return 'Could not reach the server: $error';
}

class ApiClient {
  ApiClient({http.Client? client, this.baseUrl = apiBaseUrl})
    : _client = client ?? http.Client();

  final http.Client _client;
  final String baseUrl;

  // The reranking round-trip can take tens of seconds; give it room.
  static const Duration _recommendTimeout = Duration(seconds: 240);
  static const Duration _defaultTimeout = Duration(seconds: 15);

  Future<MoviePage> fetchMovies({
    int limit = 50,
    int offset = 0,
    String? search,
  }) async {
    final params = <String, String>{
      'limit': '$limit',
      'offset': '$offset',
      if (search != null) 'search': search,
    };
    final uri = Uri.parse('$baseUrl/movies').replace(queryParameters: params);
    final response = await _client.get(uri).timeout(_defaultTimeout, onTimeout: () => throw ApiTimeoutException(_defaultTimeout));
    _throwIfNotOk(response);
    return MoviePage.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<Movie> fetchMovie(int id) async {
    final uri = Uri.parse('$baseUrl/movies/$id');
    final response = await _client.get(uri).timeout(_defaultTimeout, onTimeout: () => throw ApiTimeoutException(_defaultTimeout));
    _throwIfNotOk(response);
    return Movie.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  Future<AppUser> fetchUser(String username) async {
    final uri = Uri.parse('$baseUrl/users/$username');
    final response = await _client.get(uri).timeout(_defaultTimeout, onTimeout: () => throw ApiTimeoutException(_defaultTimeout));
    _throwIfNotOk(response);
    return AppUser.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  Future<List<Recommendation>> recommend({
    required String username,
    required String prompt,
    required double exploration,
    String? imageBase64,
  }) async {
    final uri = Uri.parse('$baseUrl/recommendations');
    final body = <String, dynamic>{
      'username': username,
      'prompt': prompt,
      'exploration': exploration,
      if (imageBase64 != null) 'image_base64': imageBase64,
    };
    final response = await _client
        .post(
          uri,
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode(body),
        )
        .timeout(_recommendTimeout, onTimeout: () => throw ApiTimeoutException(_recommendTimeout));
    _throwIfNotOk(response);
    final decoded = jsonDecode(response.body) as List;
    return decoded
        .map((item) => Recommendation.fromJson(item as Map<String, dynamic>))
        .toList();
  }

  void _throwIfNotOk(http.Response response) {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      return;
    }
    if (response.statusCode == 404) {
      throw NotFoundException(response.body);
    }
    throw ApiException(response.statusCode, response.body);
  }
}
