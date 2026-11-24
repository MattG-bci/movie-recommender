import 'dart:convert';
import 'dart:html' as html;
import 'dart:math';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;

import '../runtime_config.dart';

/// Minimal OAuth 2.0 Authorization Code + PKCE client for the hosted Flutter
/// web app. No secret is used: browser applications must be public clients.
class BrowserAuth {
  BrowserAuth._(this.accessToken);

  final String? accessToken;

  static const _tokenKey = 'entra_api_access_token';
  static const _verifierKey = 'entra_pkce_verifier';
  static const _stateKey = 'entra_oauth_state';

  static Future<BrowserAuth> bootstrap() async {
    final clientId = runtimeConfigValue('entraClientId');
    final tenantId = runtimeConfigValue('entraTenantId');
    final scope = runtimeConfigValue('apiScope');
    if (clientId.isEmpty || tenantId.isEmpty || scope.isEmpty) {
      return BrowserAuth._(null);
    }

    final uri = Uri.base;
    final code = uri.queryParameters['code'];
    if (code != null) {
      final token = await _exchangeCode(
        code: code,
        clientId: clientId,
        tenantId: tenantId,
        scope: scope,
      );
      html.window.sessionStorage[_tokenKey] = token;
      // Remove the short-lived authorization response from the address bar.
      html.window.history.replaceState(null, '', _redirectUri());
      return BrowserAuth._(token);
    }

    final cached = html.window.sessionStorage[_tokenKey];
    if (cached != null && cached.isNotEmpty) return BrowserAuth._(cached);

    _startLogin(clientId: clientId, tenantId: tenantId, scope: scope);
    return BrowserAuth._(null);
  }

  static void _startLogin({
    required String clientId,
    required String tenantId,
    required String scope,
  }) {
    final verifier = _randomUrlSafeValue();
    final state = _randomUrlSafeValue();
    html.window.sessionStorage[_verifierKey] = verifier;
    html.window.sessionStorage[_stateKey] = state;
    final challenge = base64Url
        .encode(sha256.convert(utf8.encode(verifier)).bytes)
        .replaceAll('=', '');
    final authorizationUri = Uri.https(
      'login.microsoftonline.com',
      '/$tenantId/oauth2/v2.0/authorize',
      {
        'client_id': clientId,
        'response_type': 'code',
        'redirect_uri': _redirectUri(),
        'response_mode': 'query',
        'scope': 'openid profile offline_access $scope',
        'state': state,
        'code_challenge': challenge,
        'code_challenge_method': 'S256',
      },
    );
    html.window.location.assign(authorizationUri.toString());
  }

  static Future<String> _exchangeCode({
    required String code,
    required String clientId,
    required String tenantId,
    required String scope,
  }) async {
    final expectedState = html.window.sessionStorage[_stateKey];
    if (expectedState == null ||
        expectedState != Uri.base.queryParameters['state']) {
      throw StateError(
          'The Microsoft Entra sign-in response did not match this browser session.');
    }
    final verifier = html.window.sessionStorage[_verifierKey];
    if (verifier == null)
      throw StateError('The Microsoft Entra sign-in verifier is missing.');
    final response = await http.post(
      Uri.https('login.microsoftonline.com', '/$tenantId/oauth2/v2.0/token'),
      body: {
        'client_id': clientId,
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': _redirectUri(),
        'code_verifier': verifier,
        'scope': 'openid profile offline_access $scope',
      },
    );
    html.window.sessionStorage.remove(_verifierKey);
    html.window.sessionStorage.remove(_stateKey);
    if (response.statusCode != 200) {
      throw StateError(
          'Microsoft Entra token exchange failed (${response.statusCode}).');
    }
    final payload = jsonDecode(response.body) as Map<String, dynamic>;
    final token = payload['access_token'] as String?;
    if (token == null || token.isEmpty)
      throw StateError('Microsoft Entra did not return an API access token.');
    return token;
  }

  static String _redirectUri() => '${Uri.base.origin}${Uri.base.path}';

  static String _randomUrlSafeValue() {
    final random = Random.secure();
    final bytes =
        Uint8List.fromList(List<int>.generate(32, (_) => random.nextInt(256)));
    return base64Url.encode(bytes).replaceAll('=', '');
  }
}
