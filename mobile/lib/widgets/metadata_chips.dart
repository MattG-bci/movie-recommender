import 'package:flutter/material.dart';

/// A labelled row of chips used on the movie detail screen for genres,
/// director, country and cast. There is no synopsis section anywhere.
class MetadataChipsRow extends StatelessWidget {
  const MetadataChipsRow({super.key, required this.label, required this.values});

  final String label;
  final List<String> values;

  @override
  Widget build(BuildContext context) {
    if (values.isEmpty) {
      return const SizedBox.shrink();
    }
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              for (final value in values) Chip(label: Text(value)),
            ],
          ),
        ],
      ),
    );
  }
}
