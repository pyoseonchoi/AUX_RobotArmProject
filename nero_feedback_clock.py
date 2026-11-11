"""Epoch clock matching Windows native CAN receive timestamps on Python 3.12."""

import ctypes
import sys
from time import time as _fallback_time


_FILETIME_UNIX_EPOCH = 116444736000000000
_kernel32 = None
_precise_filetime = None
if sys.platform == "win32":
    try:
        _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _precise_filetime = _kernel32.GetSystemTimePreciseAsFileTime
        _precise_filetime.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        _precise_filetime.restype = None
    except (AttributeError, OSError):
        _precise_filetime = None


def feedback_time():
    """Return epoch seconds without Python <3.13 Windows wall-clock quantization.

    Interval deadlines still use time.monotonic(). CAN timestamps keep their
    original meaning: queued frames are not restamped as newly received.
    """
    if _precise_filetime is None:
        return _fallback_time()
    # A per-call buffer also makes this safe when called by multiple threads.
    stamp = ctypes.c_uint64()
    _precise_filetime(ctypes.byref(stamp))
    return (stamp.value - _FILETIME_UNIX_EPOCH) / 10000000.0
