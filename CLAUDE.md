# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RapidShot is a high-performance Windows screen capture library using the Desktop Duplication API (DXGI). It provides ultra-fast capture (240+ FPS) with multiple backend options (NumPy, CuPy/GPU, Pillow) and supports both single-shot and continuous capture modes.

**Platform**: Windows 10+ only (relies on COM/DXGI)
**Python**: 3.8-3.11 (not 3.12+)

## Development Commands

### Setup
```bash
# Install with dev dependencies using uv (preferred)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_memory_pool.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=rapidshot --cov-report=html

# Lightweight project validation
bash scripts/validate.sh

# Include full DirectX diagnostic pass
RAPIDSHOT_VALIDATE_DIAGNOSTIC=1 bash scripts/validate.sh
```

### Code Quality
```bash
# Format code (line length: 100)
black rapidshot tests --line-length 100
isort rapidshot tests --profile black --line-length 100

# Type checking
mypy rapidshot

# Linting
flake8 rapidshot
```

### Benchmarks
```bash
# Run RapidShot benchmarks
python benchmarks/rapidshot_max_fps.py
python benchmarks/rapidshot_capture.py

# With GPU acceleration
python benchmarks/rapidshot_max_fps.py --gpu

# Compare with other libraries
python benchmarks/dxcam_max_fps.py
python benchmarks/mss_max_fps.py
```

## Architecture

### Layer Structure

The codebase follows a layered architecture from low-level COM bindings to high-level capture:

1. **`_libs/`** - COM/ctypes bindings to Windows APIs
   - `dxgi.py`: DXGI (DirectX Graphics Infrastructure) interfaces and structures
   - `d3d11.py`: Direct3D 11 interfaces for GPU operations
   - `user32.py`: User32 API bindings for cursor operations

2. **`core/`** - DXGI abstraction layer
   - `device.py`: Wraps DXGI adapter (GPU device)
   - `output.py`: Wraps DXGI output (monitor)
   - `duplicator.py`: Wraps Desktop Duplication API, handles frame acquisition and cursor capture
   - `stagesurf.py`: Manages staging surfaces for GPU-to-CPU texture copies

3. **`processor/`** - Image processing backends
   - `base.py`: Processor base class with backend selection
   - `numpy_processor.py`: CPU-based NumPy processing
   - `cupy_processor.py`: GPU-accelerated CuPy processing
   - `pillow_processor.py`: PIL-based processing (legacy)

4. **`memory_pool.py`** - Memory management
   - `NumpyMemoryPool` / `CupyMemoryPool`: Pre-allocated buffer pools
   - `PooledBuffer`: Wrapper for pooled buffers with automatic check-in/check-out
   - Critical for zero-copy operations in continuous capture mode

5. **`capture.py`** - High-level capture interface
   - `ScreenCapture`: Main user-facing class
   - Manages capture modes (single-shot vs continuous)
   - Handles error recovery and re-initialization
   - Coordinates Device, Duplicator, StageSurface, Processor, and MemoryPool

6. **`__init__.py`** - Factory and public API
   - `RapidshotFactory`: Singleton that enumerates devices/outputs and creates ScreenCapture instances
   - Public functions: `create()`, `device_info()`, `output_info()`, `clean_up()`, `reset()`

### Capture Flow

**Single-shot capture** (`grab()`):
1. User calls `screencapture.grab(region)`
2. `Duplicator.update_frame()` acquires next frame from Desktop Duplication API
3. `Device.im_context.CopySubresourceRegion()` copies region from DXGI texture to staging surface
4. `StageSurface.map()` maps GPU memory to CPU-accessible memory
5. `Processor.process()` converts BGRA to requested format (RGB/BGR/etc.)
6. Returns NumPy/CuPy array (or PooledBuffer if using memory pool)

**Continuous capture** (`start()` / `get_latest_frame()`):
1. User calls `screencapture.start(target_fps=60)`
2. Background thread continuously calls `_grab()` at target FPS
3. Frames stored in ring buffer (`deque` of `PooledBuffer` wrappers)
4. User calls `get_latest_frame()` to retrieve most recent frame from buffer
5. Memory pool ensures efficient reuse of buffers without allocation overhead

### Key Design Patterns

- **Factory Pattern**: `RapidshotFactory` centralizes device enumeration and instance creation
- **Singleton**: Only one factory instance per process
- **Resource Management**: Explicit `release()` calls required for COM objects
- **Error Recovery**: `_needs_reinit` flag triggers re-initialization on DXGI errors (device lost, access lost)
- **Memory Pooling**: Pre-allocated buffers avoid malloc overhead in high-FPS capture

## Critical Concepts

### COM Pointer Ownership Rule (Important)

- `comtypes` pointer wrappers already manage COM lifetimes.
- Do not mix unmanaged manual `Release()` calls with `comtypes`-owned pointers unless you also clear/null the pointer immediately.
- Use `rapidshot.util.ctypes_helpers.release_com_ptr()` for explicit cleanup paths.
- Double-release can silently corrupt pointer state and later fail in unrelated calls.

### Desktop Duplication API Behavior

- **Frame acquisition**: `AcquireNextFrame()` blocks until screen changes or timeout
- **Frame release**: MUST call `ReleaseFrame()` before next `AcquireNextFrame()`, or it will fail
- **Timeout handling**: Timeout means no screen update, not an error (static screen)
- **Error codes**:
  - `DXGI_ERROR_ACCESS_LOST`: Display mode changed, need to recreate duplicator
  - `DXGI_ERROR_WAIT_TIMEOUT`: No screen update within timeout (normal for static screens)
  - `DXGI_ERROR_DEVICE_REMOVED`: GPU device removed/reset (rare, needs full re-init)

### Region Handling

- Regions use **screen coordinates**: `(left, top, right, bottom)`
- Must convert to **memory coordinates** based on rotation: see `region_to_memory_region()` in `capture.py:315`
- Rotation angles: 0°, 90°, 180°, 270° (from DXGI output description)
- Region validation ensures coordinates are within bounds and form positive area

### Memory Pool Usage

- Pool buffers are pre-allocated at initialization with specific shape `(height, width, 4)` for BGRA
- `grab()` uses pool only if requested region matches pool buffer shape
- Continuous mode always uses pool for performance
- `PooledBuffer` wrappers automatically return buffers to pool when released
- Pool exhaustion returns `None` instead of allocating (prevents OOM)

### Error Recovery Strategy

`ScreenCapture` implements multi-level error handling:
1. **Transient errors**: Retry immediately (e.g., timeout)
2. **Recoverable errors**: Set `_needs_reinit` flag, attempt re-initialization with exponential backoff (capture.py:278)
3. **Permanent failure**: After max attempts, set `_capture_permanently_failed` flag
4. Continuous capture thread detects permanent failure and stops gracefully

## Common Pitfalls

1. **Not releasing resources**: Always call `screencapture.release()` or use context managers (if implemented)
2. **Manual COM release misuse**: Calling `Release()` directly on `comtypes` pointers in many places can trigger finalizer-time double-release.
3. **Region coordinates**: Remember to validate and convert based on rotation
4. **Timeout interpretation**: `DXGI_ERROR_WAIT_TIMEOUT` is not an error - screen just hasn't updated
5. **Memory pool shape mismatch**: If grab region doesn't match pool shape, it bypasses pool (slower)
6. **GPU acceleration**: Requires CuPy and compatible CUDA drivers; gracefully falls back to CPU if unavailable
7. **Python version**: Python 3.12+ may have compatibility issues with dependencies

## Recent DXGI Stabilization Work (2026-02)

This repository went through a deep DXGI/COM debugging cycle to resolve factory creation side effects and downstream crashes.

Debugging trail:
- Reproduced failures in both `diagnostic_script.py` and `rapidshot.create()`.
- Confirmed that factory creation itself was often healthy, but later COM calls crashed.
- Isolated a lifetime issue by monkey-patching factory release behavior and observing stable captures.
- Traced failure mode to double-release of COM pointers (`Release()` + `comtypes` finalizer).

Implementation outcome:
- Added safe release helper: `release_com_ptr()` in `rapidshot/util/ctypes_helpers.py`.
- Replaced high-risk manual release calls in:
  - `rapidshot/util/io.py`
  - `rapidshot/core/device.py`
  - `rapidshot/core/stagesurf.py`
  - `rapidshot/core/duplicator.py`
  - `diagnostic_script.py`

Verification outcome:
- End-to-end diagnostics complete with `FAILED: 0`.
- `rapidshot.create()` and frame grabbing now pass in repeated smoke checks.

## Testing Notes

- Tests use pytest with fixtures
- `test_memory_pool.py`: Comprehensive memory pool tests (NumPy and CuPy)
- `test_region_mapping.py`: Region coordinate conversion tests
- CuPy tests skip automatically if GPU unavailable
- Benchmarks in `benchmarks/` compare against other libraries (DXCam, MSS, etc.)

## File Location Patterns

When adding new features or fixing bugs:
- **Low-level DXGI work**: Modify `_libs/` or `core/`
- **Image processing**: Add/modify `processor/` backend
- **Capture logic**: Modify `capture.py`
- **Public API changes**: Update `__init__.py` and `RapidshotFactory`
- **Error handling**: Use error classes from `util/errors.py`
- **Logging**: Use `util/logging.py` logger (configured globally)
