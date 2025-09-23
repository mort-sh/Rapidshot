"""Utility helpers for working with ctypes values."""

from __future__ import annotations

import ctypes
from typing import Optional, Union


def pointer_to_address(ptr: Union[int, ctypes.c_void_p, ctypes._Pointer]) -> Optional[int]:
    """Return the integer address represented by *ptr*.

    Parameters
    ----------
    ptr:
        A ctypes pointer-like object or an integer address.

    Returns
    -------
    Optional[int]
        The integer address, or ``None`` if *ptr* does not represent a
        valid address.
    """

    if ptr is None:
        return None

    if isinstance(ptr, int):
        return ptr

    if isinstance(ptr, ctypes.c_void_p):
        return ptr.value

    try:
        value = ctypes.cast(ptr, ctypes.c_void_p).value
        if value is not None:
            return value
    except (TypeError, ValueError):
        pass

    if hasattr(ptr, "value") and ptr.value is not None:
        return ptr.value

    if hasattr(ptr, "contents"):
        try:
            return ctypes.addressof(ptr.contents)
        except (TypeError, ValueError):
            return None

    return None
