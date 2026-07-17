"""Process-lifetime promotion of the loaded PoPS host image for native plugins."""
from __future__ import annotations

import ctypes
import os
import sys
from threading import Lock
from typing import Any


_LOCK = Lock()
_GLOBAL_HANDLES: dict[str, Any] = {}


def ensure_native_host_global(module: Any) -> None:
    """Promote the exact loaded extension before compiling or loading a POSIX plugin.

    Generated components intentionally share the Kokkos/MPI runtimes already owned by ``_pops``.
    Keeping the returned ``CDLL`` for process lifetime makes the promotion and its reference count
    stable; a temporary handle must not be garbage-collected back to local visibility.
    """
    if sys.platform == "win32":
        return
    path = os.path.realpath(str(getattr(module, "__file__", "") or ""))
    if not path or not os.path.isfile(path):
        raise RuntimeError(
            "native component compilation/loading requires the exact loaded pops._pops image")
    with _LOCK:
        if path in _GLOBAL_HANDLES:
            return
        mode = int(getattr(os, "RTLD_NOW", 2)) | int(getattr(os, "RTLD_GLOBAL", ctypes.RTLD_GLOBAL))
        try:
            _GLOBAL_HANDLES[path] = ctypes.CDLL(path, mode=mode)
        except OSError as exc:
            raise RuntimeError(
                "failed to promote the loaded pops._pops image to RTLD_GLOBAL before native "
                "component compilation/loading: %s" % exc) from exc


__all__ = ["ensure_native_host_global"]
