# Changelog

All notable changes to GAVEL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Unit tests for `gavel.config`, `gavel.models`, and `gavel.evaluation`
- GitHub Actions CI pipeline
- Dockerfile for containerized deployment
- CONTRIBUTING.md guide
- This CHANGELOG

## [0.1.0] - 2026-01-01

### Added
- Initial release accompanying ICLR 2026 paper
- Core `gavel` package with modular architecture
- `TopicRNN` model for Cognitive Element detection
- Training pipeline (`scripts/train.py`)
- Unified evaluation pipeline (`scripts/evaluate.py`)
- Calibration using Youden's J-statistic
- Configuration system with JSON support
- Jupyter notebooks for interactive exploration
- MIT License

### Documentation
- Comprehensive README with quick start guide
- Hardware requirements specification
- CITATION.cff for proper attribution
