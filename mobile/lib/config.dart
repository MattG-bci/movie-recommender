/// App-wide configuration read from `--dart-define` flags so the API host
/// can be swapped per environment (emulator, physical device, staging, web)
/// without rebuilding source.
///
/// Example:
///   flutter run --dart-define=API_BASE_URL=http://192.168.1.20:8080
///
import 'runtime_config.dart';

/// The web container writes its API URL into `config.js` at startup. Native
/// builds use `--dart-define`, which keeps their target API configurable.
final String apiBaseUrl = runtimeConfigValue('apiBaseUrl').isNotEmpty
    ? runtimeConfigValue('apiBaseUrl')
    : const String.fromEnvironment(
        'API_BASE_URL',
        defaultValue: 'http://localhost:8080',
      );

/// Bundle identifier used for `flutter create` (iOS + Android only).
const String appBundleId = 'com.movierecommender.mobile';
