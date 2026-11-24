import 'dart:html' as html;
import 'dart:js_util' as js_util;

String runtimeConfigValue(String key) {
  final config = js_util.getProperty<Object?>(html.window, '__APP_CONFIG__');
  if (config == null) return '';
  return js_util.getProperty<Object?>(config, key)?.toString() ?? '';
}
