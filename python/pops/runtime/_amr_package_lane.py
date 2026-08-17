"""Stage the RuntimeInstance package lane required by AMR native packages.

``AmrSystem`` native-package install refuses to materialize without an active package-assembly
lane. ``pops.bind`` already stages that lane through the boundary-authority seam. Low-level
``add_equation(CompiledModel)`` callers must hit the same seam before ``_install_native_block``.
"""

from __future__ import annotations

from typing import Any


def ensure_native_block_state_route(runtime: Any, name: str, compiled: Any) -> None:
    """Install one block/state identity when the low-level add_equation path skipped pops.bind."""
    native = getattr(runtime, "_s", runtime)
    install = getattr(native, "_install_block_state_route", None)
    if not callable(install):
        return
    owner = getattr(compiled, "consumer_owner_qid", None)
    if not isinstance(owner, str) or not owner:
        owner = "pops.runtime.package.%s" % name
    identity = "%s/state" % owner
    try:
        install(name, identity)
    except Exception as error:
        message = str(error)
        if "duplicate" in message or "unique" in message:
            return
        if "must be installed before" in message:
            return
        raise


def ensure_amr_native_package_lane(runtime: Any, model: Any) -> Any:
    """Install the owned AMR package lane when the native engine does not already have one."""
    from pops._native_selector import selected_native_module
    from pops._platform_contracts import (
        ExecutionContext,
        ExecutionResource,
        PlatformManifest,
        artifact_platform_manifest,
        validate_launch,
    )
    from pops.codegen.loader import CompiledModel
    from pops.codegen._native_mpi import native_mpi_communicator
    from pops.codegen.toolchain import loader_native_dimension
    from pops.runtime._platform_manifest import native_device_resource
    from pops.runtime._platform_manifest import native_runtime_backend
    from pops.runtime._platform_manifest import native_runtime_backend_for_route

    if type(model) is not CompiledModel:
        raise TypeError("AMR package lane requires an exact CompiledModel")
    if model.target != "amr_system":
        raise ValueError("AMR package lane requires target='amr_system'")
    if model.native_dimension != loader_native_dimension():
        raise ValueError(
            "AMR package lane compiled dimension %d differs from the loaded native dimension %d"
            % (model.native_dimension, loader_native_dimension())
        )
    native = getattr(runtime, "_s", runtime)
    has_lane = getattr(native, "has_package_assembly_lane", None)
    if callable(has_lane) and has_lane():
        return getattr(runtime, "_execution_context", None)
    prepare_lane = getattr(native, "_prepare_boundary_execution_lane", None)
    if not callable(prepare_lane):
        raise TypeError("AMR package lane requires the native lane-preparation seam")

    _pops = selected_native_module(required=True)
    communicator = native_mpi_communicator(_pops)
    backend = native_runtime_backend_for_route("production", "amr_system", communicator)
    platform = artifact_platform_manifest(
        backend="production",
        target="amr_system",
        component=model,
        communicator=communicator,
        runtime_backend=backend,
    )
    if type(platform) is not PlatformManifest:
        raise TypeError("AMR package lane requires an exact PlatformManifest")
    runtime_backend = native_runtime_backend(platform)
    communicator_name = platform.communicator.require("artifact.communicator")
    datatype_name = platform.precision.storage.require("artifact.precision.storage")
    if communicator_name == "MPI_COMM_WORLD":
        from pops._native_collectives import require_world

        if datatype_name != "float64":
            raise ValueError(
                "AMR package lane has no authenticated datatype handle for %r" % datatype_name
            )
        world = require_world(_pops.mpi_world())
        context = ExecutionContext(
            backend=runtime_backend,
            communicator=ExecutionResource("communicator", "MPI_COMM_WORLD", handle=world),
            datatype=ExecutionResource("datatype", datatype_name, handle=world.datatype_float64),
            device=native_device_resource(runtime_backend),
        )
    elif communicator_name == "serial":
        context = ExecutionContext(
            backend=runtime_backend,
            communicator=ExecutionResource("communicator", "serial"),
            datatype=ExecutionResource("datatype", datatype_name),
            device=native_device_resource(runtime_backend),
        )
    else:
        raise ValueError("AMR package lane does not support communicator %r" % communicator_name)
    validate_launch(platform, context, ())
    return _install_amr_lane(runtime, context, prepare_lane)


def ensure_amr_standalone_assembly_lane(runtime: Any) -> Any:
    """Install a serial/world AMR assembly lane before low-level assembling APIs run."""
    native = getattr(runtime, "_s", runtime)
    has_lane = getattr(native, "has_package_assembly_lane", None)
    if callable(has_lane) and has_lane():
        return getattr(runtime, "_execution_context", None)
    prepare_lane = getattr(native, "_prepare_boundary_execution_lane", None)
    if not callable(prepare_lane):
        return getattr(runtime, "_execution_context", None)
    return _install_amr_lane(runtime, _standalone_amr_execution_context(), prepare_lane)


def _standalone_amr_execution_context() -> Any:
    from pops._native_selector import selected_native_module
    from pops._platform_contracts import ExecutionContext, ExecutionResource
    from pops.codegen._native_mpi import native_mpi_communicator
    from pops.runtime._platform_manifest import native_device_resource
    from pops.runtime._platform_manifest import native_runtime_backend_for_route

    module = selected_native_module(required=True)
    communicator_name = native_mpi_communicator(module)
    backend = native_runtime_backend_for_route("production", "amr_system", communicator_name)
    if communicator_name == "MPI_COMM_WORLD":
        from pops._native_collectives import require_world

        world = require_world(module.mpi_world())
        return ExecutionContext(
            backend=backend,
            communicator=ExecutionResource("communicator", "MPI_COMM_WORLD", handle=world),
            datatype=ExecutionResource("datatype", "float64", handle=world.datatype_float64),
            device=native_device_resource(backend),
        )
    if communicator_name == "serial":
        return ExecutionContext(
            backend=backend,
            communicator=ExecutionResource("communicator", "serial"),
            datatype=ExecutionResource("datatype", "float64"),
            device=native_device_resource(backend),
        )
    raise ValueError("AMR package lane does not support communicator %r" % communicator_name)


def _install_amr_lane(runtime: Any, context: Any, prepare_lane: Any) -> Any:
    from pops.runtime._component_execution_context import component_execution_data

    runtime._execution_context = context
    try:
        prepare_lane(context.communicator.handle, component_execution_data(context))
    except Exception as error:
        if "already installed" in str(error):
            return context
        raise
    return context


def ensure_system_native_package_lane(runtime: Any, compiled: Any = None) -> None:
    """Install the Uniform RuntimeInstance lane when low-level add_equation skipped pops.bind."""
    native = getattr(runtime, "_s", runtime)
    prepare = getattr(native, "_prepare_boundary_execution_lane", None)
    if not callable(prepare):
        return
    owner = getattr(compiled, "consumer_owner_qid", None)
    if not isinstance(owner, str) or not owner:
        owner = "pops.runtime.package.standalone"
    from pops._native_selector import selected_native_module
    from pops.codegen._native_mpi import native_mpi_communicator

    module = selected_native_module(required=True)
    communicator_name = native_mpi_communicator(module)
    if communicator_name == "serial":
        handle = None
    elif communicator_name == "MPI_COMM_WORLD":
        from pops._native_collectives import require_world

        handle = require_world(module.mpi_world())
    else:
        raise ValueError("System package lane does not support communicator %r" % communicator_name)
    try:
        prepare(handle, owner)
    except Exception as error:
        if "already installed" in str(error):
            return
        raise
