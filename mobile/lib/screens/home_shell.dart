import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../api/api_client.dart';
import '../services/image_source_service.dart';
import '../state/recommend_controller.dart';
import '../state/session.dart';
import 'explore_screen.dart';
import 'recommend_screen.dart';

/// The two-tab `NavigationBar` shell (Explore, Recommend) shown once signed
/// in, with a sign-out action in the app bar.
class HomeShell extends StatefulWidget {
  const HomeShell({
    super.key,
    required this.api,
    required this.images,
    required this.username,
  });

  final ApiClient api;
  final ImageSourceService images;
  final String username;

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _index = 0;
  late final RecommendController _recommendController;

  @override
  void initState() {
    super.initState();
    _recommendController = RecommendController(
      api: widget.api,
      images: widget.images,
      username: widget.username,
    );
  }

  @override
  Widget build(BuildContext context) {
    final screens = [
      ExploreScreen(api: widget.api),
      RecommendScreen(controller: _recommendController),
    ];

    return Scaffold(
      appBar: AppBar(
        title: Text(_index == 0 ? 'Explore' : 'Recommend'),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () => context.read<Session>().signOut(),
          ),
        ],
      ),
      body: IndexedStack(index: _index, children: screens),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (value) => setState(() => _index = value),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.explore_outlined), label: 'Explore'),
          NavigationDestination(icon: Icon(Icons.auto_awesome_outlined), label: 'Recommend'),
        ],
      ),
    );
  }
}
