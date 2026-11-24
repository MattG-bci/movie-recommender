/// Native builds use their own platform-specific auth flow later. Keeping this
/// no-op lets development against an unprotected local API remain simple.
class BrowserAuth {
  BrowserAuth._(this.accessToken);

  final String? accessToken;

  static Future<BrowserAuth> bootstrap() async => BrowserAuth._(null);
}
