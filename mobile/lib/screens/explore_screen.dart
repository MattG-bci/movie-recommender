import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../api/api_client.dart';
import '../state/movies_controller.dart';
import '../widgets/movie_poster.dart';
import 'movie_detail_screen.dart';

/// A responsive browse grid. Phones retain two poster columns; wider screens
/// add compact, capped cards instead of stretching two posters across the
/// viewport.
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
                    child: LayoutBuilder(
                      builder: (context, constraints) {
                        final horizontalInset =
                            ((constraints.maxWidth - 1280) / 2)
                                    .clamp(0.0, double.infinity) +
                                16;
                        return Padding(
                          padding: EdgeInsets.fromLTRB(
                              horizontalInset, 16, horizontalInset, 12),
                          child: TextField(
                            decoration: const InputDecoration(
                              hintText: 'Search movies',
                              prefixIcon: Icon(Icons.search),
                            ),
                            onChanged: controller.setSearch,
                          ),
                        );
                      },
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
                    SliverLayoutBuilder(
                      builder: (context, constraints) {
                        final horizontalInset =
                            ((constraints.crossAxisExtent - 1280) / 2)
                                    .clamp(0.0, double.infinity) +
                                16;
                        return SliverPadding(
                          padding: EdgeInsets.fromLTRB(
                              horizontalInset, 4, horizontalInset, 24),
                          sliver: SliverGrid(
                            gridDelegate:
                                const SliverGridDelegateWithMaxCrossAxisExtent(
                              maxCrossAxisExtent: 168,
                              childAspectRatio: .55,
                              crossAxisSpacing: 14,
                              mainAxisSpacing: 20,
                            ),
                            delegate: SliverChildBuilderDelegate(
                              (context, index) {
                                final movie = controller.movies[index];
                                return GestureDetector(
                                  onTap: () {
                                    Navigator.of(context).push(
                                      MaterialPageRoute(
                                        builder: (_) =>
                                            MovieDetailScreen(movie: movie),
                                      ),
                                    );
                                  },
                                  child: Hero(
                                    tag: 'movie-poster-${movie.id}',
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Expanded(
                                            child: MoviePoster(
                                                movie: movie,
                                                borderRadius: 10)),
                                        const SizedBox(height: 6),
                                        Text(
                                          movie.title,
                                          maxLines: 1,
                                          overflow: TextOverflow.ellipsis,
                                          style: Theme.of(context)
                                              .textTheme
                                              .titleSmall,
                                        ),
                                      ],
                                    ),
                                  ),
                                );
                              },
                              childCount: controller.movies.length,
                            ),
                          ),
                        );
                      },
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
