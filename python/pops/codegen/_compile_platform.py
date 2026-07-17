"""Single platform gate for every out-of-CMake shared-library compiler."""
from __future__ import annotations

import sys


def require_shared_library_compile_platform(operation: str, *, windows_supported: bool) -> None:
    if not isinstance(operation, str) or not operation:
        raise TypeError("compile operation name must be non-empty text")
    if sys.platform == "win32" and not windows_supported:
        raise NotImplementedError(
            "%s is unavailable on Windows: PoPS has no authenticated PE/COFF symbol-inspection "
            "and publication pipeline for this artifact kind. No POSIX command, .so suffix, or "
            "partially linked DLL will be emitted." % operation)


__all__ = ["require_shared_library_compile_platform"]
