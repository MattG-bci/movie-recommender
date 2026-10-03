import 'package:flutter/material.dart';

import '../api/models.dart';
import 'movie_poster.dart';

/// The `MoviePoster` widget at thumbnail size, title + year, a match-score
/// pill shown only when `matchScore != null`, and the reason in quoted
/// italics, or a muted fallback when `reason` is null.
class RecommendationCard extends StatelessWidget {
  const RecommendationCard({super.key, required this.recommendation, this.displayScore});

  final Recommendation recommendation;

  /// Pre-normalised (0-100) score to render, computed by the caller across
  /// the full result set via [normalizeMatchScores]. Falls back to the raw
  /// `matchScore` when not supplied, e.g. in isolated widget tests.
  final double? displayScore;

  @override
  Widget build(BuildContext context) {
    final movie = recommendation.movie;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(width: 72, child: MoviePoster(movie: movie)),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          '${movie.title} (${movie.releaseYear})',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                      ),
                      if (recommendation.matchScore != null) _MatchScorePill(
                        score: displayScore ?? recommendation.matchScore!,
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    recommendation.reason != null
                        ? '"${recommendation.reason}"'
                        : 'Picked by your taste profile.',
                    style: recommendation.reason != null
                        ? const TextStyle(fontStyle: FontStyle.italic)
                        : TextStyle(color: Theme.of(context).colorScheme.outline),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _MatchScorePill extends StatelessWidget {
  const _MatchScorePill({required this.score});

  final double score;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.primaryContainer,
        borderRadius: BorderRadius.circular(999),
      ),
      child: Text(
        '${score.round()}',
        style: TextStyle(color: Theme.of(context).colorScheme.onPrimaryContainer),
      ),
    );
  }
}
