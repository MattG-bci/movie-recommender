import 'package:flutter/foundation.dart';

import '../api/api_client.dart';
import '../api/models.dart';
import '../services/image_source_service.dart';

enum ImageOrigin { gallery, camera }

/// The API keeps the raw `match_score` (a CF embedding dot product, not a
/// percentage) so the client decides presentation. Min-max normalises the
/// non-null scores across the returned candidate set into a 0-100 range for
/// display; a `null` score stays `null` in the output.
List<double?> normalizeMatchScores(List<Recommendation> results) {
  final scores = results.map((r) => r.matchScore).whereType<double>().toList();
  if (scores.isEmpty) {
    return List<double?>.filled(results.length, null);
  }

  final min = scores.reduce((a, b) => a < b ? a : b);
  final max = scores.reduce((a, b) => a > b ? a : b);
  final range = max - min;

  return results.map((r) {
    final score = r.matchScore;
    if (score == null) {
      return null;
    }
    if (range == 0) {
      return 100.0;
    }
    return (score - min) / range * 100;
  }).toList();
}

/// Recommend-tab state: prompt, exploration slider, optional attached image
/// and results. `submit()` refuses to fire when both the prompt is blank and
/// no image is attached, and is a no-op while already loading.
class RecommendController extends ChangeNotifier {
  RecommendController({
    required ApiClient api,
    required ImageSourceService images,
    required this.username,
  })  : _api = api,
        _images = images;

  final ApiClient _api;
  final ImageSourceService _images;
  final String username;

  String _prompt = '';
  double _exploration = 0.3;
  String? _imageBase64;
  bool _isLoading = false;
  Object? _error;
  List<Recommendation> _results = [];

  String get prompt => _prompt;
  set prompt(String value) => _prompt = value;

  double get exploration => _exploration;
  set exploration(double value) => _exploration = value.clamp(0.0, 1.0).toDouble();

  String? get imageBase64 => _imageBase64;
  bool get isLoading => _isLoading;
  Object? get error => _error;
  List<Recommendation> get results => List.unmodifiable(_results);

  Future<void> pickImage(ImageOrigin origin) async {
    final base64 = origin == ImageOrigin.gallery
        ? await _images.pickFromGallery()
        : await _images.pickFromCamera();
    if (base64 == null) {
      // User cancelled the picker; leave state untouched.
      return;
    }
    _imageBase64 = base64;
    notifyListeners();
  }

  void clearImage() {
    _imageBase64 = null;
    notifyListeners();
  }

  Future<void> submit() async {
    if (_isLoading) {
      return;
    }
    if (_prompt.trim().isEmpty && _imageBase64 == null) {
      return;
    }

    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      _results = await _api.recommend(
        username: username,
        prompt: _prompt,
        exploration: _exploration,
        imageBase64: _imageBase64,
      );
    } catch (e) {
      _error = e;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
}
