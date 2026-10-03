import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../api/api_client.dart';
import '../state/recommend_controller.dart';
import '../widgets/recommendation_card.dart';

/// One scrollable column: multiline prompt field, an attach-image row
/// (Photo library / Take photo) with an inline thumbnail and remove button
/// once attached, an exploration slider labelled "Familiar <-> Adventurous",
/// and a full-width primary button, followed by results.
class RecommendScreen extends StatefulWidget {
  const RecommendScreen({super.key, required this.controller});

  final RecommendController controller;

  @override
  State<RecommendScreen> createState() => _RecommendScreenState();
}

class _RecommendScreenState extends State<RecommendScreen> {
  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider<RecommendController>.value(
      value: widget.controller,
      child: Consumer<RecommendController>(
        builder: (context, controller, _) {
          final normalizedScores = normalizeMatchScores(controller.results);
          return Scaffold(
            body: SafeArea(
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  TextField(
                    minLines: 3,
                    maxLines: 6,
                    decoration: const InputDecoration(
                      labelText: 'What are you in the mood for?',
                    ),
                    onChanged: (value) => controller.prompt = value,
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      OutlinedButton.icon(
                        onPressed: () => controller.pickImage(ImageOrigin.gallery),
                        icon: const Icon(Icons.photo_library_outlined),
                        label: const Text('Photo library'),
                      ),
                      const SizedBox(width: 8),
                      OutlinedButton.icon(
                        onPressed: () => controller.pickImage(ImageOrigin.camera),
                        icon: const Icon(Icons.photo_camera_outlined),
                        label: const Text('Take photo'),
                      ),
                    ],
                  ),
                  if (controller.imageBase64 != null) ...[
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Image.memory(
                            base64Decode(controller.imageBase64!),
                            width: 64,
                            height: 64,
                            fit: BoxFit.cover,
                          ),
                        ),
                        const SizedBox(width: 8),
                        TextButton(
                          onPressed: controller.clearImage,
                          child: const Text('Remove'),
                        ),
                      ],
                    ),
                  ],
                  const SizedBox(height: 16),
                  Text('Familiar ↔ Adventurous'),
                  Slider(
                    value: controller.exploration,
                    onChanged: (value) => setState(() => controller.exploration = value),
                  ),
                  const SizedBox(height: 8),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      onPressed: controller.isLoading ? null : () => controller.submit(),
                      child: controller.isLoading
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Text('Get recommendations'),
                    ),
                  ),
                  if (controller.error != null) ...[
                    const SizedBox(height: 12),
                    Text(
                      describeApiError(controller.error!),
                      style: TextStyle(color: Theme.of(context).colorScheme.error),
                    ),
                  ],
                  const SizedBox(height: 24),
                  for (var i = 0; i < controller.results.length; i++)
                    RecommendationCard(
                      recommendation: controller.results[i],
                      displayScore: normalizedScores[i],
                    ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
