# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- None

### Changed
- None

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None

## [0.1.1] - 2025-06-18
- Added TrOCR integration for improved digit recognition
- Added Pytesseract baseline comparison
- Updated requirements.txt with new dependencies

## [0.1.0] - 2025-06-18
- Initial release
- Basic model architectures:
  - YOLOv8-based dial detector
  - U-Net based digit segmenter
  - ResNet18-based digit classifier
- Configuration files for model training
- Training scripts for each model component
- Evaluation and testing scripts
- Utility modules for metrics, preprocessing, and visualization
- Dependencies in requirements.txt including:
  - PyTorch and torchvision
  - Ultralytics (YOLOv8)
  - OpenCV and Pillow
  - Albumentations for data augmentation
  - TensorBoard for visualization
