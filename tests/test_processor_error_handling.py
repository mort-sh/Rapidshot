"""
Test that processor errors propagate correctly instead of being swallowed.

This addresses the issue where NumpyProcessor.process() and CupyProcessor.process()
were catching all exceptions and returning black frames, making debugging impossible.
"""
import pytest
import numpy as np
from unittest.mock import Mock


def test_numpy_processor_propagates_invalid_rect_error():
    """Test that NumpyProcessor raises ValueError for invalid rect instead of returning black frame."""
    from rapidshot.processor.numpy_processor import NumpyProcessor
    
    processor = NumpyProcessor(color_mode='RGB')
    
    # Create an invalid rect (no pBits attribute)
    invalid_rect = Mock()
    invalid_rect.pBits = None
    
    with pytest.raises(ValueError, match="Invalid rect or pBits"):
        processor.process(
            rect=invalid_rect,
            width=100,
            height=100,
            region=(0, 0, 100, 100),
            rotation_angle=0
        )


def test_numpy_processor_propagates_invalid_rect_no_attr():
    """Test that NumpyProcessor raises ValueError when rect has no pBits attribute."""
    from rapidshot.processor.numpy_processor import NumpyProcessor
    
    processor = NumpyProcessor(color_mode='RGB')
    
    # Create an invalid rect (no pBits attribute at all)
    invalid_rect = object()
    
    with pytest.raises(ValueError, match="Invalid rect or pBits"):
        processor.process(
            rect=invalid_rect,
            width=100,
            height=100,
            region=(0, 0, 100, 100),
            rotation_angle=0
        )


# Conditional CuPy tests
try:
    import cupy
    if cupy.cuda.runtime.getDeviceCount() > 0:
        cupy_available = True
    else:
        cupy_available = False
except (ImportError, Exception):
    cupy_available = False


@pytest.mark.skipif(not cupy_available, reason="CuPy not available or no GPU")
def test_cupy_processor_propagates_invalid_rect_error():
    """Test that CupyProcessor raises ValueError for invalid rect instead of returning black frame."""
    from rapidshot.processor.cupy_processor import CupyProcessor
    
    processor = CupyProcessor(color_mode='RGB')
    
    # Create an invalid rect (no pBits attribute)
    invalid_rect = Mock()
    invalid_rect.pBits = None
    
    with pytest.raises(ValueError, match="Invalid rect or pBits"):
        processor.process(
            rect=invalid_rect,
            width=100,
            height=100,
            region=(0, 0, 100, 100),
            rotation_angle=0
        )


@pytest.mark.skipif(not cupy_available, reason="CuPy not available or no GPU")
def test_cupy_processor_propagates_invalid_rect_no_attr():
    """Test that CupyProcessor raises ValueError when rect has no pBits attribute."""
    from rapidshot.processor.cupy_processor import CupyProcessor
    
    processor = CupyProcessor(color_mode='RGB')
    
    # Create an invalid rect (no pBits attribute at all)
    invalid_rect = object()
    
    with pytest.raises(ValueError, match="Invalid rect or pBits"):
        processor.process(
            rect=invalid_rect,
            width=100,
            height=100,
            region=(0, 0, 100, 100),
            rotation_angle=0
        )

