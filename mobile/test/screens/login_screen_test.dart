import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:movie_recommender_mobile/api/api_client.dart';
import 'package:movie_recommender_mobile/api/models.dart';
import 'package:movie_recommender_mobile/screens/login_screen.dart';
import 'package:movie_recommender_mobile/state/session.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

class _MockApiClient extends Mock implements ApiClient {}

Future<void> _pumpLoginScreen(WidgetTester tester, Session session) async {
  await tester.pumpWidget(
    ChangeNotifierProvider<Session>.value(
      value: session,
      child: const MaterialApp(home: LoginScreen()),
    ),
  );
}

void main() {
  late _MockApiClient api;
  late SharedPreferences prefs;
  late Session session;

  setUp(() async {
    api = _MockApiClient();
    SharedPreferences.setMockInitialValues({});
    prefs = await SharedPreferences.getInstance();
    session = Session(api: api, prefs: prefs);
  });

  testWidgets('test_loginScreen__showsInlineErrorForUnknownUser', (tester) async {
    when(() => api.fetchUser('ghost')).thenThrow(NotFoundException('not found'));
    await _pumpLoginScreen(tester, session);

    await tester.enterText(find.byType(TextField), 'ghost');
    await tester.tap(find.text('Continue'));
    await tester.pumpAndSettle();

    expect(find.textContaining('No such user'), findsOneWidget);
  });

  testWidgets('test_loginScreen__rejectsEmptyUsernameWithoutRequest', (tester) async {
    await _pumpLoginScreen(tester, session);

    await tester.tap(find.text('Continue'));
    await tester.pumpAndSettle();

    verifyNever(() => api.fetchUser(any()));
    expect(find.text('Enter a username'), findsOneWidget);
  });

  testWidgets('test_loginScreen__disablesButtonWhileLoading', (tester) async {
    when(() => api.fetchUser('alice')).thenAnswer((_) async {
      await Future<void>.delayed(const Duration(milliseconds: 100));
      return const AppUser(id: 1, username: 'alice');
    });
    await _pumpLoginScreen(tester, session);

    await tester.enterText(find.byType(TextField), 'alice');
    await tester.tap(find.text('Continue'));
    await tester.pump();

    final button = tester.widget<FilledButton>(find.byType(FilledButton));
    expect(button.onPressed, isNull);

    await tester.pumpAndSettle();
  });
}
