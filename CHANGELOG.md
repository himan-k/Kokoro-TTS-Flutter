# Changelog

## 0.1.0
- Initial release of Kokoro TTS Flutter.
- Multi-language TTS support via ONNX Runtime.
- Includes tokenizer, phonemizer, and audio synthesis utilities.
- Example and integration tests included.

## 0.1.1
- Update Malsami library version to 0.0.2

## 0.1.2
- Update Malsami library version to 0.0.3

## 0.2.1
- Fix voice loading to support file system paths in addition to assets
- Fix ONNX output handling to properly flatten nested lists, preventing 'List<dynamic>' cast errors
- Update flutter_onnxruntime dependency to ^1.5.2 for ONNX Runtime 1.22.0 support

## 0.2.0
- Add int8 model support