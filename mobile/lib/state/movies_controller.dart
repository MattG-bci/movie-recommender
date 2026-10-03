import 'dart:async';

import 'package:flutter/foundation.dart';

import '../api/api_client.dart';
import '../api/models.dart';

/// Explore-tab paging + search state. `loadNextPage` is a no-op while
/// loading or once every result has been fetched, which prevents a
/// scroll-triggered request storm — and matters more here because each page
/// triggers a server-side OMDb fan-out.
class MoviesController extends ChangeNotifier {
  MoviesController({required ApiClient api, this.pageSize = 30}) : _api = api;

  final ApiClient _api;
  final int pageSize;

  List<Movie> _movies = [];
  bool _isLoading = false;
  int _total = 0;
  String? _search;
  Object? _error;
  Timer? _debounce;

  List<Movie> get movies => List.unmodifiable(_movies);
  bool get isLoading => _isLoading;
  bool get hasMore => _movies.length < _total;
  Object? get error => _error;

  Future<void> loadFirstPage() async {
    _movies = [];
    _total = 0;
    _error = null;
    await _loadPage(offset: 0);
  }

  Future<void> loadNextPage() async {
    if (_isLoading || !hasMore) {
      return;
    }
    await _loadPage(offset: _movies.length);
  }

  Future<void> setSearch(String query) async {
    _debounce?.cancel();
    _debounce = Timer(const Duration(milliseconds: 300), () {
      _search = query.isEmpty ? null : query;
      loadFirstPage();
    });
  }

  Future<void> _loadPage({required int offset}) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      final page = await _api.fetchMovies(
        limit: pageSize,
        offset: offset,
        search: _search,
      );
      _movies = [..._movies, ...page.items];
      _total = page.total;
    } catch (e) {
      _error = e;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  @override
  void dispose() {
    _debounce?.cancel();
    super.dispose();
  }
}
