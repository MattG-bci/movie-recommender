import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

import '../api/api_client.dart';
import '../state/session.dart';

/// A single centred form: wordmark, one text field, a filled button, an
/// inline error slot and a button loading state.
class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _controller = TextEditingController();
  String? _errorText;
  bool _isSubmitting = false;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    final username = _controller.text.trim();
    if (username.isEmpty) {
      setState(() => _errorText = 'Enter a username');
      return;
    }

    setState(() {
      _isSubmitting = true;
      _errorText = null;
    });

    final session = context.read<Session>();
    try {
      await session.signIn(username);
    } on UnknownUserException {
      setState(() => _errorText = 'No such user "$username"');
    } on ApiException {
      setState(() => _errorText = 'Could not reach the server. Try again.');
    } finally {
      if (mounted) {
        setState(() => _isSubmitting = false);
      }
    }
  }

  Future<void> _importLetterboxdExport() async {
    final username = _controller.text.trim();
    if (username.isEmpty) {
      setState(() => _errorText = 'Enter your Letterboxd username first');
      return;
    }

    setState(() {
      _isSubmitting = true;
      _errorText = null;
    });
    try {
      final opened = await launchUrl(
        Uri.parse('https://letterboxd.com/data/export/'),
        mode: LaunchMode.externalApplication,
      );
      if (!opened) {
        throw StateError('Could not open Letterboxd');
      }
      if (!mounted) return;
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
      if (file == null || bytes == null) return;

      final result = await context.read<Session>().importLetterboxdExport(
            username: username,
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
    } on ApiException catch (error) {
      if (mounted) setState(() => _errorText = error.message);
    } catch (_) {
      if (mounted) {
        setState(() => _errorText = 'Could not import the Letterboxd export.');
      }
    } finally {
      if (mounted) setState(() => _isSubmitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 360),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text('Movie Recommender', style: Theme.of(context).textTheme.headlineMedium),
                  const SizedBox(height: 32),
                  TextField(
                    controller: _controller,
                    decoration: InputDecoration(
                      labelText: 'Username',
                      errorText: _errorText,
                    ),
                    enabled: !_isSubmitting,
                    onSubmitted: (_) => _submit(),
                  ),
                  const SizedBox(height: 16),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      onPressed: _isSubmitting ? null : _submit,
                      child: _isSubmitting
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Text('Continue'),
                    ),
                  ),
                  const SizedBox(height: 12),
                  TextButton.icon(
                    onPressed: _isSubmitting ? null : _importLetterboxdExport,
                    icon: const Icon(Icons.upload_file_outlined),
                    label: const Text('Import Letterboxd export'),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
