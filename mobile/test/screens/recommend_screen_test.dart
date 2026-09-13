import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:movie_recommender_mobile/api/api_client.dart';
import 'package:movie_recommender_mobile/api/models.dart';
import 'package:movie_recommender_mobile/screens/recommend_screen.dart';
import 'package:movie_recommender_mobile/services/image_source_service.dart';
import 'package:movie_recommender_mobile/state/recommend_controller.dart';

class _MockApiClient extends Mock implements ApiClient {}

class _MockImageSourceService extends Mock implements ImageSourceService {}

Movie _makeMovie() {
  return const Movie(
    id: 1,
    title: 'Some Movie',
    releaseYear: 2020,
    genres: ['Action'],
    director: 'Director',
    country: 'US',
    actors: ['Actor'],
  );
}

void main() {
  late _MockApiClient api;
  late _MockImageSourceService images;
  late RecommendController controller;

  setUp(() {
    api = _MockApiClient();
    images = _MockImageSourceService();
    controller = RecommendController(api: api, images: images, username: 'alice');
  });

  testWidgets('test_recommendScreen__rendersResultsAfterSubmit', (tester) async {
    when(() => api.recommend(
          username: any(named: 'username'),
          prompt: any(named: 'prompt'),
          exploration: any(named: 'exploration'),
          imageBase64: any(named: 'imageBase64'),
        )).thenAnswer(
      (_) async => [Recommendation(movie: _makeMovie(), reason: 'great pick', matchScore: 9.0)],
    );

    await tester.pumpWidget(MaterialApp(home: RecommendScreen(controller: controller)));
    await tester.enterText(find.byType(TextField), 'something fun');
    await tester.tap(find.text('Get recommendations'));
    await tester.pumpAndSettle();

    expect(find.textContaining('Some Movie'), findsOneWidget);
  });

  testWidgets('test_recommendScreen__showsLoadingStateWhileSubmitting', (tester) async {
    when(() => api.recommend(
          username: any(named: 'username'),
          prompt: any(named: 'prompt'),
          exploration: any(named: 'exploration'),
          imageBase64: any(named: 'imageBase64'),
        )).thenAnswer((_) async {
      await Future<void>.delayed(const Duration(milliseconds: 100));
      return <Recommendation>[];
    });

    await tester.pumpWidget(MaterialApp(home: RecommendScreen(controller: controller)));
    await tester.enterText(find.byType(TextField), 'something fun');
    await tester.tap(find.text('Get recommendations'));
    await tester.pump();

    expect(find.byType(CircularProgressIndicator), findsOneWidget);

    await tester.pumpAndSettle();
  });
}
