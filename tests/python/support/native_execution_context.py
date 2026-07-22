"""Exact native launch context for low-level ``CompiledProblem`` integration fixtures.

The public constructor intentionally accepts only the sealed artifact returned by ``pops.compile``.
Some native integration tests exercise the lower ``compile_problem`` engine seam directly; this
test-only adapter derives the same backend proof and native resource handles without widening that
public API or inventing serial evidence under an MPI-enabled runtime.
"""
from __future__ import annotations

from typing import Any


def platform_execution_context(platform: Any) -> Any:
    """Materialize the exact native resources required by one ``PlatformManifest``."""
    from pops import _pops
    from pops._platform_contracts import (
        ExecutionContext,
        ExecutionResource,
        PlatformManifest,
        validate_launch,
    )
    from pops.runtime._platform_manifest import native_runtime_backend
    from pops.runtime._platform_manifest import native_device_resource

    if type(platform) is not PlatformManifest:
        raise TypeError("native integration context requires an exact PlatformManifest")

    backend = native_runtime_backend(platform)
    communicator_name = platform.communicator.require("artifact.communicator")
    datatype_name = platform.precision.storage.require("artifact.precision.storage")
    if communicator_name == "MPI_COMM_WORLD":
        from pops._native_collectives import require_world

        if datatype_name != "float64":
            raise ValueError(
                "native MPI integration context has no authenticated datatype handle for %r"
                % datatype_name
            )
        communicator = require_world(_pops.mpi_world())
        context = ExecutionContext(
            backend=backend,
            communicator=ExecutionResource(
                "communicator", "MPI_COMM_WORLD", handle=communicator),
            datatype=ExecutionResource(
                "datatype", datatype_name, handle=communicator.datatype_float64),
            device=native_device_resource(backend),
        )
    elif communicator_name == "serial":
        context = ExecutionContext(
            backend=backend,
            communicator=ExecutionResource("communicator", "serial"),
            datatype=ExecutionResource("datatype", datatype_name),
            device=native_device_resource(backend),
        )
    else:
        raise ValueError(
            "native integration context does not support communicator %r"
            % communicator_name
        )
    validate_launch(platform, context, ())
    return context


def artifact_execution_context(artifact: Any) -> Any:
    """Return the exact native launch context for a compiled simulation artifact."""
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact

    if type(artifact) is not CompiledSimulationArtifact:
        raise TypeError(
            "native integration context requires an exact CompiledSimulationArtifact"
        )
    return platform_execution_context(artifact.platform_manifest)


def compiled_problem_execution_context(compiled: Any, *, target: str) -> Any:
    """Return an exact :class:`ExecutionContext` for a real low-level compiled Program."""
    from pops import _pops
    from pops._platform_contracts import artifact_platform_manifest
    from pops.codegen.loader import CompiledProblem
    from pops.codegen._native_mpi import native_mpi_communicator

    if type(compiled) is not CompiledProblem:
        raise TypeError("native integration context requires an exact CompiledProblem")
    if target not in {"system", "amr_system"}:
        raise ValueError("native integration context target must be system or amr_system")
    if not compiled.abi_key:
        raise RuntimeError("CompiledProblem carries no authenticated native ABI key")

    selected_communicator = native_mpi_communicator(_pops)
    platform = artifact_platform_manifest(
        backend="production",
        target=target,
        component=compiled,
        communicator=selected_communicator,
    )
    return platform_execution_context(platform)


__all__ = [
    "artifact_execution_context",
    "compiled_problem_execution_context",
    "platform_execution_context",
]
