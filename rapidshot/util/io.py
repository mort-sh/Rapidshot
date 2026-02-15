import ctypes
import logging
from typing import List
from collections import defaultdict
import comtypes
from rapidshot._libs.dxgi import (
    CreateLatestDXGIFactory,
    IDXGIFactory1,
    IDXGIFactory6,
    IDXGIAdapter1,
    IDXGIOutput1,
    DXGI_ERROR_NOT_FOUND,
    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
)
from rapidshot._libs.user32 import (
    DISPLAY_DEVICE,
    MONITORINFOEXW,
    DISPLAY_DEVICE_ACTIVE,
    DISPLAY_DEVICE_PRIMARY_DEVICE,
)

# Configure logging
logger = logging.getLogger(__name__)


def _format_hresult(hr: int) -> str:
    return f"0x{ctypes.c_uint32(hr).value:08X}"


def _is_dxgi_not_found(com_error: comtypes.COMError) -> bool:
    if not com_error.args:
        return False
    return ctypes.c_int32(com_error.args[0]).value == ctypes.c_int32(DXGI_ERROR_NOT_FOUND).value


def _create_dxgi_factory1() -> ctypes.POINTER(IDXGIFactory1):
    factory_ptr = ctypes.c_void_p()
    hr = CreateLatestDXGIFactory(ctypes.byref(IDXGIFactory1._iid_), ctypes.byref(factory_ptr))
    if ctypes.c_long(hr).value < 0 or not factory_ptr.value:
        raise RuntimeError(f"Unable to create DXGI factory. HRESULT={_format_hresult(hr)}")
    return ctypes.cast(factory_ptr, ctypes.POINTER(IDXGIFactory1))


def _enum_adapters_with_factory1(
    dxgi_factory: ctypes.POINTER(IDXGIFactory1),
) -> List[ctypes.POINTER(IDXGIAdapter1)]:
    i = 0
    p_adapters: List[ctypes.POINTER(IDXGIAdapter1)] = []
    while True:
        try:
            p_adapter = ctypes.POINTER(IDXGIAdapter1)()
            dxgi_factory.EnumAdapters1(i, ctypes.byref(p_adapter))
            if not bool(p_adapter):
                logger.warning("EnumAdapters1 returned a null adapter pointer at index %d", i)
                break
            p_adapters.append(p_adapter)
            i += 1
        except comtypes.COMError as ce:
            if _is_dxgi_not_found(ce):
                break
            raise
    return p_adapters


def _enum_adapters_with_factory6(
    dxgi_factory6: ctypes.POINTER(IDXGIFactory6),
    gpu_preference: int,
) -> List[ctypes.POINTER(IDXGIAdapter1)]:
    p_adapters: List[ctypes.POINTER(IDXGIAdapter1)] = []
    i = 0
    while True:
        try:
            adapter_void = ctypes.c_void_p()
            dxgi_factory6.EnumAdapterByGpuPreference(
                i,
                gpu_preference,
                ctypes.byref(IDXGIAdapter1._iid_),
                ctypes.byref(adapter_void),
            )
            if not adapter_void.value:
                logger.warning(
                    "EnumAdapterByGpuPreference returned a null adapter pointer at index %d", i
                )
                break
            p_adapters.append(ctypes.cast(adapter_void, ctypes.POINTER(IDXGIAdapter1)))
            i += 1
        except comtypes.COMError as ce:
            if _is_dxgi_not_found(ce):
                break
            raise
    return p_adapters


def enum_dxgi_adapters() -> List[ctypes.POINTER(IDXGIAdapter1)]:
    dxgi_factory = _create_dxgi_factory1()
    try:
        return _enum_adapters_with_factory1(dxgi_factory)
    finally:
        dxgi_factory.Release()


def enum_dxgi_adapters_with_preference(
    gpu_preference=DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
) -> List[ctypes.POINTER(IDXGIAdapter1)]:
    """
    Enumerate DXGI adapters with a preference for high performance or power efficiency.
    Falls back to standard enumeration if DXGI 1.6 is not available.
    
    Args:
        gpu_preference: DXGI_GPU_PREFERENCE value
        
    Returns:
        List of adapter pointers
    """
    dxgi_factory = _create_dxgi_factory1()
    dxgi_factory6 = None
    try:
        try:
            dxgi_factory6 = dxgi_factory.QueryInterface(IDXGIFactory6)
            p_adapters = _enum_adapters_with_factory6(dxgi_factory6, gpu_preference)
            logger.info(f"Enumerated {len(p_adapters)} adapters using DXGI 1.6 EnumAdapterByGpuPreference")
            return p_adapters

        except comtypes.COMError as ce:
            logger.info(
                "DXGI 1.6 enumeration not available (HRESULT %s), using EnumAdapters1 fallback",
                _format_hresult(ce.args[0]) if ce.args else "N/A",
            )
            return _enum_adapters_with_factory1(dxgi_factory)
    except Exception as e:
        logger.error(f"Failed to enumerate adapters with preference: {e}")
        return _enum_adapters_with_factory1(dxgi_factory)
    finally:
        if dxgi_factory6 is not None:
            dxgi_factory6.Release()
        dxgi_factory.Release()


def enum_dxgi_outputs(
    dxgi_adapter: ctypes.POINTER(IDXGIAdapter1),
) -> List[ctypes.POINTER(IDXGIOutput1)]:
    i = 0
    p_outputs = list()
    while True:
        try:
            p_output = ctypes.POINTER(IDXGIOutput1)()
            dxgi_adapter.EnumOutputs(i, ctypes.byref(p_output))
            p_outputs.append(p_output)
            i += 1
        except comtypes.COMError as ce:
            if ctypes.c_int32(DXGI_ERROR_NOT_FOUND).value == ce.args[0]:
                break
            else:
                raise ce
    return p_outputs


def get_output_metadata():
    mapping_adapter = defaultdict(list)
    adapter = DISPLAY_DEVICE()
    adapter.cb = ctypes.sizeof(adapter)
    i = 0
    # Enumerate all adapters
    while ctypes.windll.user32.EnumDisplayDevicesW(0, i, ctypes.byref(adapter), 1):
        if adapter.StateFlags & DISPLAY_DEVICE_ACTIVE != 0:
            is_primary = bool(adapter.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE)
            mapping_adapter[adapter.DeviceName] = [adapter.DeviceString, is_primary, []]
            display = DISPLAY_DEVICE()
            display.cb = ctypes.sizeof(adapter)
            j = 0
            # Enumerate Monitors
            while ctypes.windll.user32.EnumDisplayDevicesW(
                adapter.DeviceName, j, ctypes.byref(display), 0
            ):
                mapping_adapter[adapter.DeviceName][2].append(
                    (
                        display.DeviceName,
                        display.DeviceString,
                    )
                )
                j += 1
        i += 1
    return mapping_adapter


def get_monitor_name_by_handle(hmonitor):
    info = MONITORINFOEXW()
    info.cbSize = ctypes.sizeof(MONITORINFOEXW)
    if ctypes.windll.user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
        return info
    return None
