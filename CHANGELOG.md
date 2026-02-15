# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- **Major Performance Improvement**: Refactored `NumpyProcessor` to use zero-copy numpy views instead of row-by-row Python copy loop. This eliminates ~470x slower region extraction when pitch != row_bytes, significantly improving frame capture performance.
  - Removed dependency on `pointer_to_address` helper that could return `None`
  - Added validation for pitch alignment and buffer sizes
  - Improved error handling with direct ctypes address resolution
  - Inspired by BetterCam's zero-copy approach
