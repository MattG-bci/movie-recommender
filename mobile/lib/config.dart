/// App-wide configuration read from `--dart-define` flags so the API host
/// can be swapped per environment (emulator, physical device, staging)
/// without rebuilding source.
///
/// Example:
///   flutter run --dart-define=API_BASE_URL=http://192.168.1.20:8080
const String apiBaseUrl = String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'http://localhost:8080',
);

/// Bundle identifier used for `flutter create` (iOS + Android only).
const String appBundleId = 'com.movierecommender.mobile';
