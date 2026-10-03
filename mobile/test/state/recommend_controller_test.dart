import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:movie_recommender_mobile/api/api_client.dart';
import 'package:movie_recommender_mobile/api/models.dart';
import 'package:movie_recommender_mobile/services/image_source_service.dart';
import 'package:movie_recommender_mobile/state/recommend_controller.dart';

class _MockApiClient extends Mock implements ApiClient {}

class _MockImageSourceService extends Mock implements ImageSourceService {}

Movie _makeMovie(int id) {
  return Movie(
    id: id,
    title: 'Movie $id',
    releaseYear: 2020,
    genres: const ['Action'],
    director: 'Director',
    country: 'US',
    actors: const ['Actor'],
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

  test('test_submit__populatesResults', () async {
    when(() => api.recommend(
          username: 'alice',
          prompt: any(named: 'prompt'),
          exploration: any(named: 'exploration'),
          imageBase64: any(named: 'imageBase64'),
        )).thenAnswer(
      (_) async => [Recommendation(movie: _makeMovie(1), reason: 'fun', matchScore: 9.0)],
    );
    controller.prompt = 'something fun';

    await controller.submit();

    expect(controller.results.length, 1);
    expect(controller.isLoading, isFalse);
  });

  test('test_submit__blockedWhenPromptBlankAndNoImage', () async {
    controller.prompt = '   ';

    await controller.submit();

    verifyNever(() => api.recommend(
          username: any(named: 'username'),
          prompt: any(named: 'prompt'),
          exploration: any(named: 'exploration'),
          imageBase64: any(named: 'imageBase64'),
        ));
  });

  test('test_submit__allowedWhenImageAttachedAndPromptBlank', () async {
    when(() => images.pickFromGallery()).thenAnswer((_) async => 'ZmFrZQ==');
    when(() => api.recommend(
          username: any(named: 'username'),
          prompt: any(named: 'prompt'),
          exploration: any(named: 'exploration'),
          imageBase64: any(named: 'imageBase64'),
        )).thenAnswer((_) async => []);
    await controller.pickImage(ImageOrigin.gallery);
    controller.prompt = '';

    await controller.submit();

    verify(() => api.recommend(
          username: 'alice',
          prompt: '',
          exploration: any(named: 'exploration'),
          imageBase64: 'ZmFrZQ==',
        )).called(1);
  });

  test('test_submit__noOpWhileLoading', () async {
    when(() => api.recommend(
          username: any(named: 'username'),
          prompt: any(named: 'prompt'),
          exploration: any(named: 'exploration'),
          imageBase64: any(named: 'imageBase64'),
        )).thenAnswer((_) async {
      await Future<void>.delayed(const Duration(milliseconds: 50));
      return <Recommendation>[];
    });
    controller.prompt = 'fun';

    final first = controller.submit();
    final second = controller.submit();
    await Future.wait([first, second]);

    verify(() => api.recommend(
          username: any(named: 'username'),
          prompt: any(named: 'prompt'),
          exploration: any(named: 'exploration'),
          imageBase64: any(named: 'imageBase64'),
        )).called(1);
  });

  test('test_submit__exposesErrorOnApiException', () async {
    when(() => api.recommend(
          username: any(named: 'username'),
          prompt: any(named: 'prompt'),
          exploration: any(named: 'exploration'),
          imageBase64: any(named: 'imageBase64'),
        )).thenThrow(ApiException(500, 'boom'));
    controller.prompt = 'fun';

    await controller.submit();

    expect(controller.error, isA<ApiException>());
  });

  test('test_pickImage__setsImageBase64FromGallery', () async {
    when(() => images.pickFromGallery()).thenAnswer((_) async => 'ZmFrZQ==');

    await controller.pickImage(ImageOrigin.gallery);

    expect(controller.imageBase64, 'ZmFrZQ==');
  });

  test('test_pickImage__cancelledPickLeavesStateUnchanged', () async {
    when(() => images.pickFromGallery()).thenAnswer((_) async => null);

    await controller.pickImage(ImageOrigin.gallery);

    expect(controller.imageBase64, isNull);
  });

  test('test_clearImage__removesAttachedImage', () async {
    when(() => images.pickFromGallery()).thenAnswer((_) async => 'ZmFrZQ==');
    await controller.pickImage(ImageOrigin.gallery);

    controller.clearImage();

    expect(controller.imageBase64, isNull);
  });

  test('test_exploration__isClampedToUnitInterval', () {
    controller.exploration = 1.5;
    expect(controller.exploration, 1.0);

    controller.exploration = -0.5;
    expect(controller.exploration, 0.0);
  });
}
