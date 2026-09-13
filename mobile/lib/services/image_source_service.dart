import 'dart:convert';
import 'dart:io';

import 'package:image_picker/image_picker.dart';

/// Abstraction over image_picker so screens are testable without a
/// platform channel.
abstract class ImageSourceService {
  Future<String?> pickFromGallery();
  Future<String?> pickFromCamera();
}

class ImagePickerService implements ImageSourceService {
  ImagePickerService({ImagePicker? picker}) : _picker = picker ?? ImagePicker();

  final ImagePicker _picker;

  static const double _maxWidth = 1024;
  static const int _imageQuality = 80;

  @override
  Future<String?> pickFromGallery() => _pick(ImageSource.gallery);

  @override
  Future<String?> pickFromCamera() => _pick(ImageSource.camera);

  Future<String?> _pick(ImageSource source) async {
    final file = await _picker.pickImage(
      source: source,
      maxWidth: _maxWidth,
      imageQuality: _imageQuality,
    );
    if (file == null) {
      return null;
    }
    final bytes = await File(file.path).readAsBytes();
    return base64Encode(bytes);
  }
}
