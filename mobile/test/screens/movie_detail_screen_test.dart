import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:movie_recommender_mobile/api/models.dart';
import 'package:movie_recommender_mobile/screens/movie_detail_screen.dart';
import 'package:movie_recommender_mobile/widgets/movie_poster.dart';

const _movie = Movie(
  id: 'movie-1',
  title: 'A Compact Movie',
  releaseYear: 2024,
  genres: ['Drama', 'Mystery'],
  director: 'A Director',
  country: 'GB',
  actors: ['First Actor', 'Second Actor'],
  description: 'A short synopsis shown beside the compact poster.',
);

void main() {
  testWidgets('movie detail uses a compact left poster and metadata below',
      (tester) async {
    tester.view.physicalSize = const Size(390, 844);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      const MaterialApp(home: MovieDetailScreen(movie: _movie)),
    );

    // The title appears in both the fallback poster and the detail heading.
    expect(find.text('A Compact Movie'), findsNWidgets(2));
    expect(find.text('2024'), findsOneWidget);
    expect(find.textContaining('short synopsis'), findsOneWidget);
    expect(find.text('Genres'), findsOneWidget);
    expect(find.text('Director'), findsOneWidget);
    expect(find.text('Country'), findsOneWidget);
    expect(find.text('Cast'), findsOneWidget);
    expect(tester.getSize(find.byType(MoviePoster)), const Size(120, 180));
  });
}
