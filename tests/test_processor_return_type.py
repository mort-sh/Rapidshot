"""
Test to validate that processors return only an array, not a tuple.
This test ensures the fix for the issue where NumpyProcessor.process()
was returning (array, is_pooled) instead of just the array.
"""
import pytest
import numpy as np
import ctypes
from rapidshot.processor.numpy_processor import NumpyProcessor


class MockRect:
    """Mock rectangle for testing."""
    def __init__(self, width, height):
        self.Pitch = width * 4  # BGRA format
        # Create a buffer with some test data
        size = height * self.Pitch
        self.buffer = (ctypes.c_ubyte * size)()
        # Fill with some pattern
        for i in range(size):
            self.buffer[i] = i % 256
        self.pBits = ctypes.cast(self.buffer, ctypes.POINTER(ctypes.c_ubyte))


def test_numpy_processor_returns_array_not_tuple():
    """Test that NumpyProcessor.process() returns only an array, not a tuple."""
    processor = NumpyProcessor(color_mode='RGB')
    
    width, height = 100, 100
    mock_rect = MockRect(width, height)
    region = (0, 0, width, height)
    rotation_angle = 0
    
    result = processor.process(mock_rect, width, height, region, rotation_angle)
    
    # Result should be an ndarray, not a tuple
    assert isinstance(result, np.ndarray), \
        f"Expected np.ndarray, got {type(result)}. If this is a tuple, the bug is not fixed."
    
    # Should NOT be a tuple
    assert not isinstance(result, tuple), \
        "Processor should return array only, not a tuple"


def test_numpy_processor_with_pooled_buffer():
    """Test that NumpyProcessor.process() returns array when given a pooled buffer."""
    processor = NumpyProcessor(color_mode='BGR')
    
    width, height = 50, 50
    mock_rect = MockRect(width, height)
    region = (0, 0, width, height)
    rotation_angle = 0
    
    # Create a pre-allocated buffer (simulating pooled buffer)
    output_buffer = np.zeros((height, width, 4), dtype=np.uint8)
    
    result = processor.process(mock_rect, width, height, region, rotation_angle, output_buffer)
    
    # Result should be an ndarray
    assert isinstance(result, np.ndarray), \
        f"Expected np.ndarray, got {type(result)}"
    
    # Should NOT be a tuple
    assert not isinstance(result, tuple), \
        "Processor should return array only, not a tuple"


def test_numpy_processor_returns_same_buffer_when_no_shape_change():
    """Test that processor returns the same buffer object when shape doesn't change."""
    processor = NumpyProcessor(color_mode='BGRA')  # No conversion needed
    
    width, height = 50, 50
    mock_rect = MockRect(width, height)
    region = (0, 0, width, height)
    rotation_angle = 0  # No rotation
    
    # Create a pre-allocated buffer
    output_buffer = np.zeros((height, width, 4), dtype=np.uint8)
    
    result = processor.process(mock_rect, width, height, region, rotation_angle, output_buffer)
    
    # When there's no color conversion (BGRA) and no rotation,
    # the result should be the same object as output_buffer
    assert result is output_buffer, \
        "When no shape changes occur, processor should return the same buffer object"


def test_numpy_processor_returns_different_buffer_with_color_conversion():
    """Test that processor returns a different buffer when color conversion changes channels."""
    processor = NumpyProcessor(color_mode='RGB')  # Will change from 4 to 3 channels
    
    width, height = 50, 50
    mock_rect = MockRect(width, height)
    region = (0, 0, width, height)
    rotation_angle = 0
    
    # Create a pre-allocated buffer (4 channels)
    output_buffer = np.zeros((height, width, 4), dtype=np.uint8)
    
    result = processor.process(mock_rect, width, height, region, rotation_angle, output_buffer)
    
    # Since RGB conversion reduces from 4 to 3 channels,
    # the result should be a different object
    assert result is not output_buffer, \
        "When color conversion changes channel count, processor should return a different buffer"
    
    # But it should still be 3 channels (RGB)
    assert result.shape[2] == 3, \
        f"RGB conversion should produce 3-channel output, got {result.shape[2]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
