import 'package:flutter/foundation.dart';

import '../api/api_client.dart';
import '../api/models.dart';
import '../services/image_source_service.dart';

enum ImageOrigin { gallery, camera }

/// Recommend-tab state: prompt, optional attached image
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

  String prompt = '';
  String? _imageBase64;
  bool _isLoading = false;
  Object? _error;
  List<Recommendation> _results = [];

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
    if (prompt.trim().isEmpty && _imageBase64 == null) {
      return;
    }

    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      _results = await _api.recommend(
        username: username,
        prompt: prompt,
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
