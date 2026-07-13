"""Generic AOT lowering of authenticated component source packages."""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from pops._manifest_protocol import exact_mapping, strict_json_loads
from pops.external._package_data import ComponentPackageError
from pops.external.artifacts import CompiledComponentArtifact, ComponentRuntimeContract
from pops.external.packages import ExternalComponent
from pops.identity import make_identity
from pops.interfaces import NumericalFlux

from .toolchain import (
    _probe_cxx_std,
    loader_cxx_std,
    pops_header_signature,
    pops_include,
    pops_loader_build_flags,
    _run_compile,
)


_CPP_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*$")
_CPP_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IR_FIELDS = frozenset({"schema_version", "language", "source"})


def _cpp_token(value: Any, *, where: str, qualified: bool = False) -> str:
    expression = _CPP_TYPE if qualified else _CPP_NAME
    if not isinstance(value, str) or expression.fullmatch(value) is None:
        raise ComponentPackageError("entry_point", where, "invalid C++ identifier")
    return value


def _resolved_target(component: ExternalComponent) -> dict[str, Any]:
    variants = tuple(component.component_manifest.target["variants"])
    supported = [item for item in variants if (
        item["dimension"] == 2 and item["scalar"] == "float64" and item["device"] == "cpu"
    )]
    if not supported:
        raise ComponentPackageError(
            "target", "component.target",
            "current AOT route proves only dimension=2, scalar=float64, device=cpu")
    return dict(supported[0])


def _state_components(component: ExternalComponent) -> int:
    value = component.component_manifest.signature.get("state_components")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ComponentPackageError(
            "signature", "signature.state_components",
            "NumericalFlux AOT requires an exact positive state_components")
    return value


def _wrapper_source(component: ExternalComponent, symbols: dict[str, str]) -> str:
    manifest = component.component_manifest
    interface = component.component_type.interface
    if interface != NumericalFlux:
        raise NotImplementedError("AOT lowering is not implemented for %s" % interface.uri)
    if component.parameters:
        raise ComponentPackageError(
            "parameters", "parameters", "current NumericalFlux AOT route accepts no parameters")
    if manifest.requirements:
        raise ComponentPackageError(
            "requirements", "requirements",
            "current NumericalFlux AOT route has no external provider resolver")
    entries = manifest.entry_points
    header = str(entries["header"])
    if header.startswith("/") or ".." in Path(header).parts or "\\" in header:
        raise ComponentPackageError("entry_point", "entry_points.header", "unsafe header path")
    component_type = _cpp_token(entries["component"], where="entry_points.component", qualified=True)
    flux = _cpp_token(entries["numerical_flux"], where="entry_points.numerical_flux")
    stability = _cpp_token(entries["stability_bound"], where="entry_points.stability_bound")
    count = _state_components(component)
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


def _materialize(package: Any, root: Path) -> list[str]:
    sources = []
    for payload in package.payloads:
        destination = root / payload.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = payload.content
        if payload.kind == "ir":
            try:
                ir = exact_mapping(
                    strict_json_loads(content, where=payload.path), _IR_FIELDS, where=payload.path)
            except (TypeError, ValueError) as exc:
                raise ComponentPackageError("ir", payload.path, str(exc)) from exc
            if ir["schema_version"] != 1 or ir["language"] != "c++" \
                    or not isinstance(ir["source"], str):
                raise ComponentPackageError(
                    "ir", payload.path, "expected pops C++ IR schema_version=1")
            destination = destination.with_suffix(".cpp")
            content = ir["source"].encode("utf-8")
            sources.append(str(destination))
        elif payload.kind == "source":
            sources.append(str(destination))
        destination.write_bytes(content)
    return sources


def compile_component(component: ExternalComponent, *, cxx: Any = None,
                      include: Any = None) -> CompiledComponentArtifact:
    """Instantiate, compile, link and audit one source component for the proved CPU target."""
    if type(component) is not ExternalComponent:
        raise TypeError("compile_component requires an exact ExternalComponent")
    target = _resolved_target(component)
    package = component.component_type.package
    include = include or pops_include()
    signature = pops_header_signature(include)
    compiler, cflags, lflags = pops_loader_build_flags(cxx)
    standard = _probe_cxx_std(compiler, loader_cxx_std())
    abi = "%s|%s|%s" % (signature, compiler, standard)
    from pops.runtime.platform_manifest import proven_serial_manifest
    platform_manifest = proven_serial_manifest(
        backend="aot-component", target="component", abi=abi)
    component.component_manifest.require_target(target)
    stem = make_identity("component-instantiation", {
        "package": package.identity.to_data(),
        "component": component.component_manifest.component_id,
    }).hexdigest[:16]
    symbols = {
        name: "pops_component_%s_%s" % (stem, name)
        for name in component.component_type.interface.runtime_entry_points
    }
    source = _wrapper_source(component, symbols)
    with tempfile.TemporaryDirectory(prefix="pops-component-") as temporary:
        root = Path(temporary)
        package_sources = _materialize(package, root)
        wrapper = root / "pops_component_wrapper.cpp"
        wrapper.write_text(source, encoding="utf-8")
        output = root / ("component" + (".dylib" if sys.platform == "darwin" else ".so"))
        command = [compiler, "-shared", "-fPIC", "-std=" + standard, *cflags,
                   "-I", include, "-I", str(root), str(wrapper), *package_sources,
                   "-o", str(output), *lflags]
        _run_compile(command, "compile external component package")
        binary = output.read_bytes()
    from pops.external.packages import _binary_identity
    artifact = CompiledComponentArtifact(
        component_id=component.component_manifest.component_id,
        component_manifest=component.component_manifest.manifest_digest,
        runtime_contract=ComponentRuntimeContract.from_manifest(component.component_manifest),
        interface=component.component_type.interface, platform_manifest=platform_manifest,
        entry_symbols=symbols, binary_identity=_binary_identity(binary), binary=binary,
        source_package=package.identity, fixed_signature=False,
        suffix=".dylib" if sys.platform == "darwin" else ".so")
    artifact.verify()
    return artifact


__all__ = ["compile_component"]
