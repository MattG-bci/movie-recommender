import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../api/api_client.dart';

const _usernameKey = 'username';

/// Thrown by [Session.signIn] when the username does not exist.
class UnknownUserException implements Exception {
  UnknownUserException(this.username);

  final String username;

  @override
  String toString() => 'UnknownUserException($username)';
}

/// Identification-only session: no passwords, no tokens. A username is
/// validated against `GET /users/{username}` and cached on-device so a
/// returning user skips the network round trip.
class Session extends ChangeNotifier {
  Session({required ApiClient api, required SharedPreferences prefs})
      : _api = api,
        _prefs = prefs;

  final ApiClient _api;
  final SharedPreferences _prefs;

  String? _username;

  String? get username => _username;
  bool get isSignedIn => _username != null;

  Future<void> restore() async {
    _username = _prefs.getString(_usernameKey);
    notifyListeners();
  }

  Future<void> signIn(String username) async {
    final trimmed = username.trim();
    try {
      final user = await _api.fetchUser(trimmed);
      await _prefs.setString(_usernameKey, user.username);
      _username = user.username;
      notifyListeners();
    } on NotFoundException {
      throw UnknownUserException(trimmed);
    }
  }

  Future<void> signOut() async {
    await _prefs.remove(_usernameKey);
    _username = null;
    notifyListeners();
  }
}
