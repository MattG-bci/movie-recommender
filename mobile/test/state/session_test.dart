import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:movie_recommender_mobile/api/api_client.dart';
import 'package:movie_recommender_mobile/api/models.dart';
import 'package:movie_recommender_mobile/state/session.dart';
import 'package:shared_preferences/shared_preferences.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  late _MockApiClient api;
  late SharedPreferences prefs;

  setUp(() async {
    api = _MockApiClient();
    SharedPreferences.setMockInitialValues({});
    prefs = await SharedPreferences.getInstance();
  });

  test('test_signIn__persistsUsernameOnSuccess', () async {
    when(() => api.fetchUser('alice')).thenAnswer(
      (_) async => const AppUser(id: 1, username: 'alice'),
    );
    final session = Session(api: api, prefs: prefs);

    await session.signIn('alice');

    expect(session.username, 'alice');
    expect(session.isSignedIn, isTrue);
    expect(prefs.getString('username'), 'alice');
  });

  test('test_signIn__throwsUnknownUserExceptionOn404', () async {
    when(() => api.fetchUser('ghost')).thenThrow(NotFoundException('not found'));
    final session = Session(api: api, prefs: prefs);

    expect(
      () => session.signIn('ghost'),
      throwsA(isA<UnknownUserException>()),
    );
  });

  test('test_signIn__doesNotPersistOnFailure', () async {
    when(() => api.fetchUser('ghost')).thenThrow(NotFoundException('not found'));
    final session = Session(api: api, prefs: prefs);

    try {
      await session.signIn('ghost');
    } on UnknownUserException {
      // expected
    }

    expect(prefs.getString('username'), isNull);
    expect(session.isSignedIn, isFalse);
  });

  test('test_restore__loadsPersistedUsername', () async {
    SharedPreferences.setMockInitialValues({'username': 'bob'});
    prefs = await SharedPreferences.getInstance();
    final session = Session(api: api, prefs: prefs);

    await session.restore();

    expect(session.username, 'bob');
    expect(session.isSignedIn, isTrue);
  });

  test('test_signOut__clearsPersistedUsername', () async {
    when(() => api.fetchUser('alice')).thenAnswer(
      (_) async => const AppUser(id: 1, username: 'alice'),
    );
    final session = Session(api: api, prefs: prefs);
    await session.signIn('alice');

    await session.signOut();

    expect(session.isSignedIn, isFalse);
    expect(prefs.getString('username'), isNull);
  });
}
