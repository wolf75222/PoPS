"""Versioned external-component contracts over ComponentManifest v2 facets.

An interface is immutable protocol metadata. It names exact small-interface bindings and package
entry points; builtins and external components therefore cross the same manifest trust boundary.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class NativeInterfaceBackend(Protocol):
    """Narrow ABI provider consumed by the generic component toolchain."""

    def resolve_target(self, component: Any) -> Mapping[str, Any]: ...

    def wrapper_source(self, component: Any, symbols: Mapping[str, str]) -> str: ...

    def bind_installed(self, component: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class ComponentInterface:
    uri: str
    version: int
    bindings: tuple[tuple[str, str], ...]
    entry_points: tuple[str, ...]
    runtime_entry_points: tuple[str, ...]
    native_backend: NativeInterfaceBackend | None = field(
        default=None, repr=False, compare=False)

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
        backend = self.native_backend
        if backend is not None and not isinstance(backend, NativeInterfaceBackend):
            raise TypeError(
                "native interface backend must implement resolve_target(), wrapper_source(), "
                "and bind_installed()")

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

    def resolve_native_target(self, component: Any) -> dict[str, Any]:
        backend = self.native_backend
        if backend is None:
            raise NotImplementedError(
                "interface %s@%d has no installed native ABI provider"
                % (self.uri, self.version))
        return dict(backend.resolve_target(component))

    def emit_native_wrapper(self, component: Any, symbols: Mapping[str, str]) -> str:
        backend = self.native_backend
        if backend is None:
            raise NotImplementedError(
                "interface %s@%d has no native wrapper provider"
                % (self.uri, self.version))
        source = backend.wrapper_source(component, symbols)
        if not isinstance(source, str) or not source.strip():
            raise TypeError("native interface backend returned an empty wrapper source")
        return source

    def bind_installed(self, component: Any) -> Any:
        backend = self.native_backend
        if backend is None:
            raise NotImplementedError(
                "interface %s@%d has no installed binding provider"
                % (self.uri, self.version))
        return backend.bind_installed(component)


_CPP_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*$")
_CPP_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _cpp_token(value: Any, *, where: str, qualified: bool = False) -> str:
    from pops.external._package_data import ComponentPackageError

    expression = _CPP_TYPE if qualified else _CPP_NAME
    if not isinstance(value, str) or expression.fullmatch(value) is None:
        raise ComponentPackageError("entry_point", where, "invalid C++ identifier")
    return value


class _NumericalFluxNativeBackend:
    """CPU/double ABI provider for the NumericalFlux interface."""

    __slots__ = ()

    def resolve_target(self, component: Any) -> Mapping[str, Any]:
        from pops.external._package_data import ComponentPackageError

        variants = tuple(component.component_manifest.target["variants"])
        supported = [item for item in variants if (
            item["dimension"] == 2
            and item["scalar"] == "float64"
            and item["device"] == "cpu"
        )]
        if not supported:
            raise ComponentPackageError(
                "target", "component.target",
                "NumericalFlux ABI proves only dimension=2, scalar=float64, device=cpu")
        return dict(supported[0])

    def wrapper_source(self, component: Any, symbols: Mapping[str, str]) -> str:
        from pops.external._package_data import ComponentPackageError

        manifest = component.component_manifest
        if component.parameters:
            raise ComponentPackageError(
                "parameters", "parameters",
                "NumericalFlux CPU ABI accepts no runtime component parameters")
        if manifest.requirements:
            raise ComponentPackageError(
                "requirements", "requirements",
                "NumericalFlux CPU ABI requires every provider to be resolved before AOT")
        entries = manifest.entry_points
        header = str(entries["header"])
        if header.startswith("/") or ".." in Path(header).parts or "\\" in header:
            raise ComponentPackageError(
                "entry_point", "entry_points.header", "unsafe header path")
        component_type = _cpp_token(
            entries["component"], where="entry_points.component", qualified=True)
        flux = _cpp_token(
            entries["numerical_flux"], where="entry_points.numerical_flux")
        stability = _cpp_token(
            entries["stability_bound"], where="entry_points.stability_bound")
        count = manifest.signature.get("state_components")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ComponentPackageError(
                "signature", "signature.state_components",
                "NumericalFlux CPU ABI requires an exact positive state_components")
        return f'''#include <pops/runtime/program/component_package.hpp>
#include "{header}"
#include <cstddef>

namespace generated {{
using Scalar = double;
inline constexpr std::size_t components = {count};
using Trace = ::pops::component_package::Trace<Scalar, components>;
using Face = ::pops::component_package::Face<Scalar, 2>;
using Providers = ::pops::component_package::NumericalFluxProviders<Scalar, components>;
using Component = {component_type};

inline Trace trace(const double* values) {{
  Trace result{{}};
  for (std::size_t i = 0; i < components; ++i) result.values[i] = values[i];
  return result;
}}
inline Face face(const double* normal) {{
  Face result{{}};
  result.normal[0] = normal[0]; result.normal[1] = normal[1];
  return result;
}}
}}  // namespace generated

extern "C" int {symbols['numerical_flux']}(
    const double* left, const double* right, const double* normal, double* output) {{
  if (!left || !right || !normal || !output) return 2;
  try {{
    const auto value = generated::Component{{}}.{flux}(
        generated::trace(left), generated::trace(right), generated::face(normal),
        generated::Providers{{}});
    for (std::size_t i = 0; i < generated::components; ++i) output[i] = value[i];
    return 0;
  }} catch (...) {{ return 1; }}
}}

extern "C" int {symbols['stability_bound']}(
    const double* left, const double* right, const double* normal, double* output) {{
  if (!left || !right || !normal || !output) return 2;
  try {{
    *output = generated::Component{{}}.{stability}(
        generated::trace(left), generated::trace(right), generated::face(normal),
        generated::Providers{{}});
    return 0;
  }} catch (...) {{ return 1; }}
}}
'''

    def bind_installed(self, component: Any) -> Any:
        from pops.external.artifacts import NumericalFluxCpuBinding

        return NumericalFluxCpuBinding(component)


NumericalFlux = ComponentInterface(
    "pops://interfaces/numerical-flux",
    1,
    (("lowering", "numerical_flux"), ("stability", "stability_bound")),
    ("header", "component", "numerical_flux", "stability_bound"),
    ("numerical_flux", "stability_bound"),
    _NumericalFluxNativeBackend(),
)


__all__ = ["NativeInterfaceBackend", "ComponentInterface", "NumericalFlux"]
