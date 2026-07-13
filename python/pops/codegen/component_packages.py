"""Generic AOT lowering of authenticated component source packages."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

from pops._manifest_protocol import exact_mapping, strict_json_loads
from pops.external._package_data import ComponentPackageError
from pops.external.artifacts import CompiledComponentArtifact, ComponentRuntimeContract
from pops.external.packages import ExternalComponent
from pops.identity import make_identity

from .toolchain import (
    _probe_cxx_std,
    loader_cxx_std,
    pops_header_signature,
    pops_include,
    pops_loader_build_flags,
    _run_compile,
)


_IR_FIELDS = frozenset({"schema_version", "language", "source"})


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
    interface = component.component_type.interface
    target = interface.resolve_native_target(component)
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
    source = interface.emit_native_wrapper(component, symbols)
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
