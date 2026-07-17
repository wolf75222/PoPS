"""Package version from the generated release contract.

The generated value derives from ``project(VERSION)`` and is available without importing native
code. A built extension is authenticated against it when native execution is requested; pure
authoring never degrades to an ambiguous ``"unknown"`` version.
"""
from __future__ import annotations

from ._generated_release_contract import PACKAGE_VERSION


__version__ = PACKAGE_VERSION


def authenticate_native_version(native: object) -> None:
    actual = getattr(native, "__version__", None)
    if actual != PACKAGE_VERSION:
        raise ImportError(
            "pops native/package version mismatch: native=%r, package=%r"
            % (actual, PACKAGE_VERSION)
        )
