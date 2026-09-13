import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api/api_client.dart';
import 'screens/home_shell.dart';
import 'screens/login_screen.dart';
import 'services/image_source_service.dart';
import 'state/session.dart';
import 'theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final prefs = await SharedPreferences.getInstance();
  final api = ApiClient();
  runApp(MovieRecommenderApp(api: api, prefs: prefs));
}

class MovieRecommenderApp extends StatelessWidget {
  const MovieRecommenderApp({
    super.key,
    required this.api,
    required this.prefs,
    ImageSourceService? images,
  }) : _images = images;

  final ApiClient api;
  final SharedPreferences prefs;
  final ImageSourceService? _images;

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider<Session>(
      create: (_) => Session(api: api, prefs: prefs)..restore(),
      child: MaterialApp(
        title: 'Movie Recommender',
        theme: buildAppTheme(Brightness.light),
        darkTheme: buildAppTheme(Brightness.dark),
        home: _RootScreen(api: api, images: _images ?? ImagePickerService()),
      ),
    );
  }
}

class _RootScreen extends StatelessWidget {
  const _RootScreen({required this.api, required this.images});

  final ApiClient api;
  final ImageSourceService images;

  @override
  Widget build(BuildContext context) {
    final session = context.watch<Session>();
    if (!session.isSignedIn) {
      return const LoginScreen();
    }
    return HomeShell(api: api, images: images, username: session.username!);
  }
}
