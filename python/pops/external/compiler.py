"""Public AOT compilation of authenticated external component packages."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pops._manifest_protocol import exact_mapping, strict_json_loads

from ._package_data import ComponentPackageError
from .artifacts import CompiledComponentArtifact, ComponentRuntimeContract
from .packages import ExternalComponent


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


def compile_component(
    component: ExternalComponent,
    *,
    cxx: Any = None,
    include: Any = None,
) -> CompiledComponentArtifact:
    """Instantiate, compile, link and audit one source component for the proved CPU target."""
    if type(component) is not ExternalComponent:
        raise TypeError("compile_component requires an exact ExternalComponent")
    from pops.codegen._compile_platform import require_shared_library_compile_platform
    require_shared_library_compile_platform("compile_component", windows_supported=False)

    # Toolchain ownership stays in the private codegen implementation. Importing the public
    # external-component package remains pure until the explicit compilation operation is called.
    from pops.codegen.toolchain import (
        _probe_cxx_std,
        _run_compile,
        _check_headers_match_module,
        loader_cxx_std,
        pops_include,
        pops_loader_build_flags,
    )

    interface = component.component_type.interface
    target = interface.resolve_native_target(component)
    package = component.component_type.package
    include = include or pops_include()
    signature = _check_headers_match_module(include)
    compiler, cflags, lflags = pops_loader_build_flags(cxx)
    cflags = [*cflags, '-DPOPS_HEADER_SIG="%s"' % signature]
    standard = _probe_cxx_std(compiler, loader_cxx_std())
    from pops import _pops
    from pops.codegen._native_mpi import native_mpi_communicator
    from pops._platform_contracts import artifact_platform_manifest
    from pops.runtime._platform_manifest import native_runtime_backend_for_route

    # The component is compiled with the same shared loader flags as generated Programs.  Its
    # manifest must therefore describe that selected host communicator as well; claiming ``serial``
    # for a binary built with POPS_HAS_MPI defeats the exact launch gate later at installation.
    communicator = native_mpi_communicator(_pops)
    runtime_backend = native_runtime_backend_for_route(
        "aot-component", "component", communicator)
    runtime_device = runtime_backend.device.require("runtime.device")
    normalized_runtime_device = "cpu" if runtime_device in ("host", "cpu") else runtime_device
    if target["device"] != normalized_runtime_device:
        raise ComponentPackageError(
            "target", "component.target",
            "component target device %r differs from installed Kokkos target %r"
            % (target["device"], normalized_runtime_device))
    host_abi = getattr(_pops, "abi_key", None)
    if not callable(host_abi) or not isinstance((abi := host_abi()), str) or not abi:
        raise RuntimeError("loaded pops._pops exposes no exact native abi_key()")
    platform_manifest = artifact_platform_manifest(
        backend="aot-component",
        target="component",
        component=SimpleNamespace(abi_key=abi),
        communicator=communicator,
        runtime_backend=runtime_backend,
    )
    component.component_manifest.require_target(target)
    symbols = {
        name: component.component_manifest.entry_points[name]
        for name in component.component_type.interface.runtime_entry_points
    }
    with tempfile.TemporaryDirectory(prefix="pops-component-") as temporary:
        root = Path(temporary)
        package_sources = _materialize(package, root)
        if not package_sources:
            raise ComponentPackageError(
                "source",
                "payloads",
                "native ABI packages require at least one source or C++ IR translation unit",
            )
        output = root / ("component" + (".dylib" if sys.platform == "darwin" else ".so"))
        command = [
            compiler,
            "-shared",
            "-fPIC",
            "-std=" + standard,
            *cflags,
            "-I",
            include,
            "-I",
            str(root),
            *package_sources,
            "-o",
            str(output),
            *lflags,
        ]
        _run_compile(command, "compile external component package")
        binary = output.read_bytes()
    from .packages import _binary_identity

    artifact = CompiledComponentArtifact(
        component_id=component.component_manifest.component_id,
        component_manifest=component.component_manifest.manifest_digest,
        runtime_contract=ComponentRuntimeContract.from_manifest(component.component_manifest),
        interface=component.component_type.interface,
        platform_manifest=platform_manifest,
        entry_symbols=symbols,
        binary_identity=_binary_identity(binary),
        binary=binary,
        source_package=package.identity,
        fixed_signature=False,
        suffix=".dylib" if sys.platform == "darwin" else ".so",
    )
    artifact.verify()
    return artifact


__all__ = ["compile_component"]
