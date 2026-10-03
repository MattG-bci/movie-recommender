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
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            pinned: true,
            expandedHeight: 360,
            flexibleSpace: FlexibleSpaceBar(
              background: Hero(
                tag: 'movie-poster-${movie.id}',
                child: MoviePoster(movie: movie, borderRadius: 0),
              ),
            ),
          ),
          SliverPadding(
            padding: const EdgeInsets.all(20),
            sliver: SliverList(
              delegate: SliverChildListDelegate([
                Text(
                  '${movie.title} (${movie.releaseYear})',
                  style: Theme.of(context).textTheme.headlineMedium,
                ),
                const SizedBox(height: 16),
                MetadataChipsRow(label: 'Genres', values: movie.genres),
                MetadataChipsRow(label: 'Director', values: [movie.director]),
                MetadataChipsRow(label: 'Country', values: [movie.country]),
                MetadataChipsRow(label: 'Cast', values: movie.actors),
              ]),
            ),
          ),
        ],
      ),
    );
  }
}
