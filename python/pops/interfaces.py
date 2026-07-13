"""Versioned component interfaces consumed by generic package lowering.

Interfaces are immutable protocol identities, not concrete component categories.  A compiler may
implement an interface once and then instantiate any conforming external component without adding a
component-specific branch or binding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ComponentInterface:
    uri: str
    version: int
    entry_points: tuple[str, ...]
    runtime_entry_points: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri.startswith("pops://interfaces/"):
            raise ValueError("component interface URI must use pops://interfaces/")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("component interface version must be an integer >= 1")
        names = tuple(self.entry_points)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise TypeError("component interface entry points must be non-empty strings")
        if len(names) != len(set(names)):
            raise ValueError("component interface entry points must be unique")
        object.__setattr__(self, "entry_points", names)
        runtime = tuple(self.runtime_entry_points)
        if not runtime or any(name not in names for name in runtime):
            raise ValueError("runtime entry points must be a non-empty subset of entry points")
        object.__setattr__(self, "runtime_entry_points", runtime)

    @property
    def key(self) -> tuple[str, int]:
        return self.uri, self.version

    def to_data(self) -> dict[str, Any]:
        return {"uri": self.uri, "version": self.version}

    def manifest_ref(self) -> tuple[str, int]:
        """Canonical ComponentManifest row (the manifest owns set normalization)."""
        return self.uri, self.version

    def require_manifest(self, manifest: Any, *, source_package: bool = True) -> None:
        rows = tuple(manifest.interfaces)
        if self.manifest_ref() not in rows:
            raise ValueError(
                "component %r does not implement interface %s@%d"
                % (manifest.component_id, self.uri, self.version))
        required = self.entry_points if source_package else self.runtime_entry_points
        missing = [name for name in required if name not in manifest.entry_points]
        if missing:
            raise ValueError(
                "component %r is missing %s interface entry point(s): %s"
                % (manifest.component_id, self.uri, ", ".join(missing)))


NumericalFlux = ComponentInterface(
    "pops://interfaces/numerical-flux", 1,
    ("header", "component", "numerical_flux", "stability_bound"),
    ("numerical_flux", "stability_bound"),
)


__all__ = ["ComponentInterface", "NumericalFlux"]
