"""Generated, versioned native component interfaces.

The catalog closes the ABI vocabulary while component implementations remain open.  This module is
data only: it contains no scientific dispatcher, Python FFI backend, or component-specific binding.
Source conformers and fixed binaries export the same audited C table getter and are loaded by the
native runtime once during installation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pops._generated_component_interfaces import (
    NATIVE_COMPONENT_ABI_VERSION,
    NATIVE_COMPONENT_CATALOG_SHA256,
    NATIVE_COMPONENT_INTERFACES,
)


_TABLE_ENTRY_POINT = "interface_table"
_TABLE_SYMBOL = "pops_component_interface_v1"


@dataclass(frozen=True, slots=True)
class ComponentInterface:
    """One exact generated C/POD table contract."""

    name: str
    abi_id: int
    uri: str
    version: int
    cpp_table: str
    hot_path: bool
    facets: tuple[str, ...]
    operations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("component interface name must be non-empty")
        if isinstance(self.abi_id, bool) or not isinstance(self.abi_id, int) or self.abi_id < 0:
            raise TypeError("component interface ABI id must be a non-negative integer")
        if not isinstance(self.uri, str) or not self.uri.startswith("pops://interfaces/"):
            raise ValueError("component interface URI must use pops://interfaces/")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("component interface version must be an integer >= 1")
        if not isinstance(self.cpp_table, str) or not self.cpp_table:
            raise TypeError("component interface C table must be non-empty")
        if not isinstance(self.hot_path, bool):
            raise TypeError("component interface hot_path must be boolean")
        facets = tuple(self.facets)
        operations = tuple(self.operations)
        if not facets or len(facets) != len(set(facets)):
            raise ValueError("component interface facets must be unique and non-empty")
        if not operations or len(operations) != len(set(operations)):
            raise ValueError("component interface operations must be unique and non-empty")
        object.__setattr__(self, "facets", facets)
        object.__setattr__(self, "operations", operations)

    @property
    def key(self) -> tuple[str, int]:
        return self.uri, self.version

    @property
    def table_entry_point(self) -> str:
        return _TABLE_ENTRY_POINT

    @property
    def table_symbol(self) -> str:
        return _TABLE_SYMBOL

    @property
    def runtime_entry_points(self) -> tuple[str, ...]:
        return (_TABLE_ENTRY_POINT,)

    @property
    def entry_points(self) -> tuple[str, ...]:
        return self.runtime_entry_points

    def manifest_declarations(self) -> tuple[dict[str, str], ...]:
        """Exact schema-v2 rows; every facet resolves through the one table getter."""
        return tuple({"name": facet, "mode": "entry_point", "binding": _TABLE_ENTRY_POINT}
                     for facet in self.facets)

    def signature_declaration(self) -> dict[str, Any]:
        return {
            "id": self.abi_id,
            "name": self.name,
            "uri": self.uri,
            "version": self.version,
            "catalog_sha256": NATIVE_COMPONENT_CATALOG_SHA256,
            "protocol_abi": NATIVE_COMPONENT_ABI_VERSION,
            "cpp_table": self.cpp_table,
            "hot_path": self.hot_path,
            "operations": tuple(self.operations),
        }

    def to_data(self) -> dict[str, Any]:
        return self.signature_declaration()

    def require_manifest(self, manifest: Any, *, source_package: bool = True) -> None:
        del source_package  # Source and fixed packages implement the identical table contract.
        rows = tuple(manifest.interfaces)
        if any(not isinstance(row, Mapping) for row in rows):
            raise TypeError("ComponentManifest interfaces must use v2 binding rows")
        actual = {(row["name"], row["mode"], row["binding"]) for row in rows}
        expected = {(facet, "entry_point", _TABLE_ENTRY_POINT) for facet in self.facets}
        if actual != expected or set(manifest.facets) != set(self.facets):
            raise ValueError(
                "component %r does not implement exact interface %s@%d"
                % (manifest.component_id, self.uri, self.version))
        native = manifest.signature.get("native_interface")
        expected_signature = self.signature_declaration()
        if not isinstance(native, Mapping) or dict(native) != expected_signature:
            raise ValueError(
                "component %r does not carry the generated native interface identity %s@%d"
                % (manifest.component_id, self.uri, self.version))
        if set(manifest.entry_points) != {_TABLE_ENTRY_POINT}:
            raise ValueError(
                "component %r must expose only the authenticated native interface table getter"
                % manifest.component_id)
        if manifest.entry_points[_TABLE_ENTRY_POINT] != _TABLE_SYMBOL:
            raise ValueError(
                "component %r must export %s" % (manifest.component_id, _TABLE_SYMBOL))

    def resolve_native_target(self, component: Any) -> dict[str, Any]:
        """Select the sole fixed v1 POD target; ambiguity is refused before compilation."""
        from pops.external._package_data import ComponentPackageError

        variants = tuple(component.component_manifest.target["variants"])
        supported = [dict(row) for row in variants
                     if row["scalar"] == "float64" and row["device"] == "cpu"]
        if len(supported) != 1:
            raise ComponentPackageError(
                "target", "component.target",
                "native component ABI v1 requires one exact float64 CPU target variant")
        return supported[0]


def _interface(row: Mapping[str, Any]) -> ComponentInterface:
    return ComponentInterface(
        name=row["name"], abi_id=row["id"], uri=row["uri"], version=row["version"],
        cpp_table=row["cpp_table"], hot_path=row["hot_path"], facets=tuple(row["facets"]),
        operations=tuple(row["operations"]),
    )


_BY_NAME = MappingProxyType({row["name"]: _interface(row)
                             for row in NATIVE_COMPONENT_INTERFACES})


def resolve(name: str) -> ComponentInterface:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            "unknown native component interface %r (valid: %s)"
            % (name, ", ".join(_BY_NAME))) from None


NumericalFlux = resolve("numerical_flux")
GhostBoundary = resolve("ghost_boundary")
FieldBoundaryClosure = resolve("field_boundary_closure")
Tagger = resolve("tagger")
Clustering = resolve("clustering")
Transfer = resolve("transfer")
Reflux = resolve("reflux")
FieldSolver = resolve("field_solver")
Writer = resolve("writer")
FieldTopology = resolve("field_topology")


__all__ = [
    "ComponentInterface", "resolve", "NumericalFlux", "GhostBoundary",
    "FieldBoundaryClosure", "Tagger", "Clustering", "Transfer", "Reflux",
    "FieldSolver", "Writer", "FieldTopology",
]
