"""Versioned external-component contracts over ComponentManifest v2 facets.

An interface is immutable protocol metadata. It names exact small-interface bindings and package
entry points; builtins and external components therefore cross the same manifest trust boundary.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ComponentInterface:
    uri: str
    version: int
    bindings: tuple[tuple[str, str], ...]
    entry_points: tuple[str, ...]
    runtime_entry_points: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri.startswith("pops://interfaces/"):
            raise ValueError("component interface URI must use pops://interfaces/")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("component interface version must be an integer >= 1")
        bindings = tuple(self.bindings)
        if not bindings or any(
                not isinstance(row, tuple) or len(row) != 2
                or any(not isinstance(value, str) or not value for value in row)
                for row in bindings):
            raise TypeError("component interface bindings must be (facet, entry_point) pairs")
        if len({facet for facet, _ in bindings}) != len(bindings):
            raise ValueError("component interface facets must be unique")
        object.__setattr__(self, "bindings", bindings)
        names = tuple(self.entry_points)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise TypeError("component interface entry points must be non-empty strings")
        if len(names) != len(set(names)):
            raise ValueError("component interface entry points must be unique")
        bound_entries = {binding for _, binding in bindings}
        if not bound_entries.issubset(names):
            raise ValueError("component interface bindings must name declared entry points")
        object.__setattr__(self, "entry_points", names)
        runtime = tuple(self.runtime_entry_points)
        if not runtime or not set(runtime).issubset(bound_entries):
            raise ValueError(
                "runtime entry points must be a non-empty subset of bound facet entry points")
        object.__setattr__(self, "runtime_entry_points", runtime)

    @property
    def key(self) -> tuple[str, int]:
        return self.uri, self.version

    @property
    def facets(self) -> tuple[str, ...]:
        return tuple(facet for facet, _ in self.bindings)

    def manifest_declarations(self) -> tuple[dict[str, str], ...]:
        """Exact ComponentManifest v2 interface rows for this external contract."""
        return tuple({"name": facet, "mode": "entry_point", "binding": binding}
                     for facet, binding in self.bindings)

    def to_data(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "version": self.version,
            "bindings": [
                {"facet": facet, "entry_point": binding}
                for facet, binding in self.bindings
            ],
        }

    def require_manifest(self, manifest: Any, *, source_package: bool = True) -> None:
        rows = tuple(manifest.interfaces)
        if any(not isinstance(row, Mapping) for row in rows):
            raise TypeError("ComponentManifest interfaces must use v2 binding rows")
        actual = {
            (row["name"], row["mode"], row["binding"])
            for row in rows
        }
        expected = {
            (facet, "entry_point", binding) for facet, binding in self.bindings
        }
        if actual != expected or set(manifest.facets) != set(self.facets):
            raise ValueError(
                "component %r does not implement exact interface %s@%d"
                % (manifest.component_id, self.uri, self.version))
        required = self.entry_points if source_package else self.runtime_entry_points
        missing = [name for name in required if name not in manifest.entry_points]
        if missing:
            raise ValueError(
                "component %r is missing %s interface entry point(s): %s"
                % (manifest.component_id, self.uri, ", ".join(missing)))


NumericalFlux = ComponentInterface(
    "pops://interfaces/numerical-flux",
    1,
    (("lowering", "numerical_flux"), ("stability", "stability_bound")),
    ("header", "component", "numerical_flux", "stability_bound"),
    ("numerical_flux", "stability_bound"),
)


__all__ = ["ComponentInterface", "NumericalFlux"]
