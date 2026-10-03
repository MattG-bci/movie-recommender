import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';

import '../api/models.dart';

/// Renders `movie.posterUrl` via [CachedNetworkImage] when present.
/// Otherwise — or when the network image fails to load — falls back to a
/// designed typographic tile: a two-stop gradient deterministically derived
/// from `movie.id` (so the same film looks the same across launches and
/// scroll recycling) with the title set in the display typeface. OMDb will
/// legitimately miss films, so this is not decoration: it is what keeps a
/// partially-covered grid looking finished rather than broken.
class MoviePoster extends StatelessWidget {
  const MoviePoster({super.key, required this.movie, this.borderRadius = 16});

  final Movie movie;
  final double borderRadius;

  static const List<List<Color>> _gradients = [
    [Color(0xFF6750A4), Color(0xFF9A82DB)],
    [Color(0xFF386A20), Color(0xFF6DBE45)],
    [Color(0xFFB3261E), Color(0xFFE46962)],
    [Color(0xFF00696F), Color(0xFF4FD8E0)],
    [Color(0xFF984061), Color(0xFFE893AF)],
    [Color(0xFF7C5800), Color(0xFFF2BB4A)],
  ];

  List<Color> get _fallbackColors => _gradients[movie.id.abs() % _gradients.length];

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(borderRadius),
      child: AspectRatio(
        aspectRatio: 2 / 3,
        child: movie.posterUrl != null
            ? CachedNetworkImage(
                imageUrl: movie.posterUrl!,
                fit: BoxFit.cover,
                errorWidget: (context, url, error) => _FallbackTile(
                  title: movie.title,
                  colors: _fallbackColors,
                ),
                placeholder: (context, url) => _FallbackTile(
                  title: movie.title,
                  colors: _fallbackColors,
                ),
              )
            : _FallbackTile(title: movie.title, colors: _fallbackColors),
      ),
    );
  }
}

class _FallbackTile extends StatelessWidget {
  const _FallbackTile({required this.title, required this.colors});

  final String title;
  final List<Color> colors;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: colors,
        ),
      ),
      padding: const EdgeInsets.all(12),
      alignment: Alignment.center,
      child: Text(
        title,
        textAlign: TextAlign.center,
        maxLines: 4,
        overflow: TextOverflow.ellipsis,
        style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.white),
      ),
    );
  }
}
