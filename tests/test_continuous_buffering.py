import collections
from threading import Event, Lock

import numpy as np
import pytest

pytest.importorskip("comtypes")

from rapidshot.capture import ScreenCapture
from rapidshot.memory_pool import NumpyMemoryPool, PooledBuffer


def make_capture(max_buffer_len=2):
    capture = object.__new__(ScreenCapture)
    capture.max_buffer_len = max_buffer_len
    capture._pooled_frames_deque = collections.deque(maxlen=max_buffer_len)
    capture._capture_lock = Lock()
    capture._frame_available_event = Event()
    capture._stop_capture_event = Event()
    capture._capture_thread = None
    capture._timer_handle = None
    capture._frame_count = 0
    capture.is_capturing = False
    capture.nvidia_gpu = False
    capture.memory_pool = None
    return capture


def test_append_frame_entry_evicts_and_releases_pooled_buffer():
    capture = make_capture(max_buffer_len=2)
    pool = NumpyMemoryPool(buffer_shape=(2, 2, 4), dtype=np.uint8, num_buffers=2)
    capture.memory_pool = pool

    pooled_one = pool.checkout()
    pooled_two = pool.checkout()
    capture._append_frame_entry(pooled_one)
    capture._append_frame_entry(pooled_two)

    stats = pool.get_stats()
    assert stats["available"] == 0
    assert stats["in_use"] == 2

    capture._append_frame_entry(np.zeros((2, 2, 3), dtype=np.uint8))

    stats = pool.get_stats()
    assert stats["available"] == 1
    assert stats["in_use"] == 1
    assert capture._pooled_frames_deque[0] is pooled_two
    assert isinstance(capture._pooled_frames_deque[1], np.ndarray)

    capture.stop()
    stats = pool.get_stats()
    assert stats["available"] == 2
    assert stats["in_use"] == 0
    pool.destroy_pool()


def test_get_latest_frame_returns_non_pooled_numpy_array():
    capture = make_capture()
    frame = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))
    capture._append_frame_entry(frame)

    latest = capture.get_latest_frame()
    assert isinstance(latest, np.ndarray)
    assert np.array_equal(latest, frame)


def test_get_latest_frame_returns_pooled_buffer_array():
    capture = make_capture()
    pool = NumpyMemoryPool(buffer_shape=(2, 2, 4), dtype=np.uint8, num_buffers=1)
    capture.memory_pool = pool
    pooled = pool.checkout()
    pooled.array.fill(9)
    capture._append_frame_entry(pooled)

    latest = capture.get_latest_frame()
    assert isinstance(latest, np.ndarray)
    assert np.array_equal(latest, pooled.array)

    capture.stop()
    pool.destroy_pool()


def test_duplicate_frame_entry_creates_new_numpy_copy():
    capture = make_capture()
    frame = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))

    duplicated = capture._duplicate_frame_entry(frame)
    assert isinstance(duplicated, np.ndarray)
    assert duplicated is not frame
    assert np.array_equal(duplicated, frame)


def test_duplicate_frame_entry_from_pooled_uses_pool():
    capture = make_capture()
    pool = NumpyMemoryPool(buffer_shape=(2, 2, 4), dtype=np.uint8, num_buffers=2)
    capture.memory_pool = pool
    pooled = pool.checkout()
    pooled.array.fill(3)

    duplicated = capture._duplicate_frame_entry(pooled)
    assert isinstance(duplicated, PooledBuffer)
    assert duplicated is not pooled
    assert np.array_equal(duplicated.array, pooled.array)

    pooled.release()
    duplicated.release()
    pool.destroy_pool()
