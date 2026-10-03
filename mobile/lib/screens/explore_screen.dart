import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../api/api_client.dart';
import '../state/movies_controller.dart';
import '../widgets/movie_poster.dart';
import 'movie_detail_screen.dart';

/// A `CustomScrollView` with a pinned search field and a two-column
/// `SliverGrid` of poster tiles at 2:3 aspect ratio, plus a first-load
/// skeleton, an empty state for no search results, and a retry state on
/// error.
class ExploreScreen extends StatefulWidget {
  const ExploreScreen({super.key, required this.api});

  final ApiClient api;

  @override
  State<ExploreScreen> createState() => _ExploreScreenState();
}

class _ExploreScreenState extends State<ExploreScreen> {
  late final MoviesController _controller;
  late final ScrollController _scrollController;

  @override
  void initState() {
    super.initState();
    _controller = MoviesController(api: widget.api);
    _scrollController = ScrollController()..addListener(_onScroll);
    _controller.loadFirstPage();
  }

  void _onScroll() {
    if (_scrollController.position.pixels >
        _scrollController.position.maxScrollExtent - 400) {
      _controller.loadNextPage();
    }
  }

  @override
  void dispose() {
    _scrollController.dispose();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider<MoviesController>.value(
      value: _controller,
      child: Scaffold(
        body: SafeArea(
          child: Consumer<MoviesController>(
            builder: (context, controller, _) {
              return CustomScrollView(
                controller: _scrollController,
                slivers: [
                  SliverToBoxAdapter(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: TextField(
                        decoration: const InputDecoration(
                          hintText: 'Search movies',
                          prefixIcon: Icon(Icons.search),
                        ),
                        onChanged: controller.setSearch,
                      ),
                    ),
                  ),
                  if (controller.error != null)
                    SliverFillRemaining(
                      child: _RetryState(
                        onRetry: controller.loadFirstPage,
                        error: controller.error,
                      ),
                    )
                  else if (controller.movies.isEmpty && controller.isLoading)
                    const SliverFillRemaining(
                      child: Center(child: CircularProgressIndicator()),
                    )
                  else if (controller.movies.isEmpty)
                    const SliverFillRemaining(
                      child: Center(child: Text('No movies found')),
                    )
                  else
                    SliverPadding(
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      sliver: SliverGrid(
                        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: 2,
                          childAspectRatio: 2 / 3,
                          crossAxisSpacing: 12,
                          mainAxisSpacing: 12,
                        ),
                        delegate: SliverChildBuilderDelegate(
                          (context, index) {
                            final movie = controller.movies[index];
                            return GestureDetector(
                              onTap: () {
                                Navigator.of(context).push(
                                  MaterialPageRoute(
                                    builder: (_) => MovieDetailScreen(movie: movie),
                                  ),
                                );
                              },
                              child: Hero(
                                tag: 'movie-poster-${movie.id}',
                                child: Column(
                                  children: [
                                    Expanded(child: MoviePoster(movie: movie)),
                                    const SizedBox(height: 4),
                                    Text(
                                      movie.title,
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                  ],
                                ),
                              ),
                            );
                          },
                          childCount: controller.movies.length,
                        ),
                      ),
                    ),
                ],
              );
            },
          ),
        ),
      ),
    );
  }
}

class _RetryState extends StatelessWidget {
  const _RetryState({required this.onRetry, this.error});

  final VoidCallback onRetry;
  final Object? error;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            error == null ? 'Something went wrong.' : describeApiError(error!),
            textAlign: TextAlign.center,
            style: TextStyle(color: Theme.of(context).colorScheme.error),
          ),
          const SizedBox(height: 12),
          FilledButton(onPressed: onRetry, child: const Text('Retry')),
        ],
      ),
    );
  }
}
