"""Internal native execution layer.

Importing the package is inert. Individual engine symbols load ``_pops`` only when explicitly
requested by compilation/binding/execution; pure planning modules under this package stay usable
without an extension.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "Profile": ("pops.runtime.profile", "Profile"),
    "PerformanceSummary": ("pops.runtime.profile", "PerformanceSummary"),
    "RuntimeInspectionReport": ("pops.runtime.inspection", "RuntimeInspectionReport"),
    "numerical_defaults_report": ("pops.runtime.defaults", "numerical_defaults_report"),
    "CapabilityProof": ("pops.runtime.platform_manifest", "CapabilityProof"),
    "PrecisionPolicy": ("pops.runtime.platform_manifest", "PrecisionPolicy"),
    "PlatformManifest": ("pops.runtime.platform_manifest", "PlatformManifest"),
    "RuntimeBackendManifest": ("pops.runtime.platform_manifest", "RuntimeBackendManifest"),
    "ExecutionResource": ("pops.runtime.platform_manifest", "ExecutionResource"),
    "ExecutionContext": ("pops.runtime.platform_manifest", "ExecutionContext"),
    "FieldViewDescriptor": ("pops.runtime.platform_manifest", "FieldViewDescriptor"),
    "PlatformContractError": ("pops.runtime.platform_manifest", "PlatformContractError"),
    "validate_launch": ("pops.runtime.platform_manifest", "validate_launch"),
    "launch_checked": ("pops.runtime.platform_manifest", "launch_checked"),
    "ConsumerGraph": ("pops.runtime.consumer", "ConsumerGraph"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError("module 'pops.runtime' has no public attribute %r" % name)
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
