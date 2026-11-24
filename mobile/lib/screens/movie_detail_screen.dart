import 'package:flutter/material.dart';

import '../api/models.dart';
import '../widgets/metadata_chips.dart';
import '../widgets/movie_poster.dart';

class MovieDetailScreen extends StatelessWidget {
  const MovieDetailScreen({super.key, required this.movie});

  final Movie movie;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(),
      body: SafeArea(
        top: false,
        child: LayoutBuilder(
          builder: (context, constraints) {
            final posterWidth = constraints.maxWidth >= 700 ? 180.0 : 120.0;
            final horizontalPadding = constraints.maxWidth >= 700
                ? (constraints.maxWidth - 640) / 2
                : 20.0;
            return SingleChildScrollView(
              padding: EdgeInsets.fromLTRB(
                  horizontalPadding, 16, horizontalPadding, 32),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      SizedBox(
                        width: posterWidth,
                        child: Hero(
                          tag: 'movie-poster-${movie.id}',
                          child: MoviePoster(movie: movie, borderRadius: 10),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(movie.title,
                                style:
                                    Theme.of(context).textTheme.headlineSmall),
                            const SizedBox(height: 4),
                            Text(
                              '${movie.releaseYear}',
                              style: Theme.of(context)
                                  .textTheme
                                  .titleMedium
                                  ?.copyWith(
                                    color: Theme.of(context)
                                        .colorScheme
                                        .onSurfaceVariant,
                                  ),
                            ),
                            if (movie.description != null &&
                                movie.description!.isNotEmpty) ...[
                              const SizedBox(height: 12),
                              Text(movie.description!),
                            ],
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),
                  MetadataChipsRow(label: 'Genres', values: movie.genres),
                  MetadataChipsRow(label: 'Director', values: [movie.director]),
                  MetadataChipsRow(label: 'Country', values: [movie.country]),
                  MetadataChipsRow(label: 'Cast', values: movie.actors),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}
