import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:movie_recommender_mobile/api/models.dart';
import 'package:movie_recommender_mobile/widgets/recommendation_card.dart';

Movie _makeMovie() {
  return const Movie(
    id: 'movie-1',
    title: 'Some Movie',
    releaseYear: 2020,
    genres: ['Action'],
    director: 'Director',
    country: 'US',
    actors: ['Actor'],
  );
}

void main() {
  testWidgets('test_recommendationCard__showsMatchScorePill', (tester) async {
    final recommendation = Recommendation(
        movie: _makeMovie(), reason: 'great pick', matchScore: 8.7);

    await tester.pumpWidget(
      MaterialApp(home: RecommendationCard(recommendation: recommendation)),
    );

    expect(find.text('8.7/10'), findsOneWidget);
  });

  testWidgets('test_recommendationCard__hidesMatchScorePillWhenNull',
      (tester) async {
    final recommendation = Recommendation(
        movie: _makeMovie(), reason: 'great pick', matchScore: null);

    await tester.pumpWidget(
      MaterialApp(home: RecommendationCard(recommendation: recommendation)),
    );

    expect(find.text('8.7/10'), findsNothing);
  });

  testWidgets('test_recommendationCard__showsFallbackTextWhenReasonIsNull',
      (tester) async {
    final recommendation =
        Recommendation(movie: _makeMovie(), reason: null, matchScore: 5.0);

    await tester.pumpWidget(
      MaterialApp(home: RecommendationCard(recommendation: recommendation)),
    );

    expect(find.text('Picked by your taste profile.'), findsOneWidget);
  });
}
