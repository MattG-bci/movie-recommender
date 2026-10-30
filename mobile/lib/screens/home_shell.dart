import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

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

  Future<void> _importLetterboxdExport() async {
    try {
      final opened = await launchUrl(
        Uri.parse('https://letterboxd.com/data/export/'),
        mode: LaunchMode.externalApplication,
      );
      if (!opened || !mounted) return;

      final chooseFile = await showDialog<bool>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Download your Letterboxd export'),
          content: const Text(
            'In Letterboxd, request and download your data export. Return here when the ZIP is ready; we will read it automatically.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Not yet'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Choose export'),
            ),
          ],
        ),
      );
      if (chooseFile != true) return;

      final selection = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: const ['zip', 'csv'],
        withData: true,
      );
      final files = selection?.files;
      final file = files != null && files.length == 1 ? files.single : null;
      final bytes = file?.bytes;
      if (file == null || bytes == null || !mounted) return;

      final result = await context.read<Session>().importLetterboxdExport(
            username: widget.username,
            bytes: bytes,
            filename: file.name,
          );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Imported ${result.importedRatings} of ${result.ratedRows} ratings'
            '${result.unmatchedRows == 0 ? '' : ' (${result.unmatchedRows} not in our catalogue)'}.',
          ),
        ),
      );
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content: Text('Could not import the Letterboxd export.')),
        );
      }
    }
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
            icon: const Icon(Icons.upload_file_outlined),
            tooltip: 'Import Letterboxd export',
            onPressed: _importLetterboxdExport,
          ),
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
          NavigationDestination(
              icon: Icon(Icons.explore_outlined), label: 'Explore'),
          NavigationDestination(
              icon: Icon(Icons.auto_awesome_outlined), label: 'Recommend'),
        ],
      ),
    );
  }
}
