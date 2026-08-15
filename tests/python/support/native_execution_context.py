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
    from pops.runtime._platform_manifest import native_runtime_backend_for_route

    if type(compiled) is not CompiledProblem:
        raise TypeError("native integration context requires an exact CompiledProblem")
    if target not in {"system", "amr_system"}:
        raise ValueError("native integration context target must be system or amr_system")
    if not compiled.abi_key:
        raise RuntimeError("CompiledProblem carries no authenticated native ABI key")

    selected_communicator = native_mpi_communicator(_pops)
    runtime_backend = native_runtime_backend_for_route(
        "production", target, selected_communicator)
    platform = artifact_platform_manifest(
        backend="production",
        target=target,
        component=compiled,
        communicator=selected_communicator,
        runtime_backend=runtime_backend,
    )
    return platform_execution_context(platform)


def install_compiled_model_amr_test_lane(runtime: Any, model: Any) -> Any:
    """Install the exact AMR package lane required by a low-level test fixture.

    This is deliberately narrower than ``pops.bind``: the caller retains its
    low-level ``AmrSystem`` fixture, while this helper derives the same native
    execution context from its exact detached ``CompiledModel`` and installs
    the owned RuntimeInstance lane before any package can materialize.
    """
    from pops import _pops
    from pops._platform_contracts import artifact_platform_manifest
    from pops.codegen.loader import CompiledModel
    from pops.codegen._native_mpi import native_mpi_communicator
    from pops.runtime._component_execution_context import component_execution_data
    from pops.runtime._platform_manifest import native_runtime_backend_for_route

    if type(model) is not CompiledModel:
        raise TypeError("AMR test lane requires an exact CompiledModel")
    if model.target != "amr_system" or model.native_dimension != 2:
        raise ValueError("AMR test lane requires one exact Dim=2 amr_system CompiledModel")
    native = getattr(runtime, "_s", runtime)
    prepare_lane = getattr(native, "_prepare_boundary_execution_lane", None)
    if not callable(prepare_lane):
        raise TypeError("AMR test lane requires the native lane-preparation seam")

    communicator = native_mpi_communicator(_pops)
    backend = native_runtime_backend_for_route("production", "amr_system", communicator)
    platform = artifact_platform_manifest(
        backend="production",
        target="amr_system",
        component=model,
        communicator=communicator,
        runtime_backend=backend,
    )
    context = platform_execution_context(platform)
    runtime._execution_context = context
    prepare_lane(context.communicator.handle, component_execution_data(context))
    return context


__all__ = [
    "artifact_execution_context",
    "compiled_problem_execution_context",
    "install_compiled_model_amr_test_lane",
    "platform_execution_context",
]
