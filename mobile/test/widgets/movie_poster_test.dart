import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:movie_recommender_mobile/api/models.dart';
import 'package:movie_recommender_mobile/widgets/movie_poster.dart';

Movie _makeMovie({int id = 1, String title = 'Some Movie', String? posterUrl}) {
  return Movie(
    id: id,
    title: title,
    releaseYear: 2020,
    genres: const ['Action'],
    director: 'Director',
    country: 'US',
    actors: const ['Actor'],
    posterUrl: posterUrl,
  );
}

BoxDecoration? _fallbackDecoration(WidgetTester tester) {
  final container = tester.widget<Container>(
    find.descendant(of: find.byType(MoviePoster), matching: find.byType(Container)),
  );
  return container.decoration as BoxDecoration?;
}

void main() {
  testWidgets('test_moviePoster__rendersNetworkImageWhenPosterUrlPresent', (tester) async {
    final movie = _makeMovie(posterUrl: 'http://example.com/p.jpg');

    await tester.pumpWidget(MaterialApp(home: MoviePoster(movie: movie)));

    expect(find.byType(CachedNetworkImage), findsOneWidget);
    final image = tester.widget<CachedNetworkImage>(find.byType(CachedNetworkImage));
    expect(image.imageUrl, 'http://example.com/p.jpg');
  });

  testWidgets('test_moviePoster__rendersFallbackWhenPosterUrlIsNull', (tester) async {
    final movie = _makeMovie(posterUrl: null, title: 'Fallback Title');

    await tester.pumpWidget(MaterialApp(home: MoviePoster(movie: movie)));
    await tester.pump();

    expect(find.text('Fallback Title'), findsOneWidget);
    expect(find.byType(CachedNetworkImage), findsNothing);
  });

  testWidgets('test_moviePoster__fallbackGradientIsDeterministicForSameMovieId', (tester) async {
    final movieA = _makeMovie(id: 42, title: 'Title A', posterUrl: null);
    final movieB = _makeMovie(id: 42, title: 'Title B', posterUrl: null);

    await tester.pumpWidget(MaterialApp(home: MoviePoster(movie: movieA)));
    await tester.pump();
    final decorationA = _fallbackDecoration(tester);

    await tester.pumpWidget(MaterialApp(home: MoviePoster(movie: movieB)));
    await tester.pump();
    final decorationB = _fallbackDecoration(tester);

    expect((decorationA?.gradient as LinearGradient?)?.colors,
        (decorationB?.gradient as LinearGradient?)?.colors);
  });
}
