"""Native Serial/MPI control plane for the saved-state scientific runners."""


def rank_and_size(world):
    return (0, 1) if world is None else (int(world.rank), int(world.size))


def gather_bytes(world, payload):
    if type(payload) is not bytes:
        raise TypeError("scientific control payload must be exact bytes")
    return (payload,) if world is None else tuple(world.allgather_bytes(payload))


def broadcast_bytes(world, payload):
    if type(payload) is not bytes:
        raise TypeError("scientific broadcast payload must be exact bytes")
    return payload if world is None else world.broadcast_bytes(payload, root=0)


def execution_context(artifact):
    import pops
    from pops._native_selector import selected_native_module
    from pops.codegen._native_mpi import native_mpi_communicator
    from pops.runtime._platform_manifest import execution_context_for_bind
    from pops.runtime_environment import runtime_environment_report

    artifact.verify()
    native = selected_native_module(required=True)
    route = native_mpi_communicator(native)
    report = runtime_environment_report()
    if report.get("communicator") != route:
        raise RuntimeError("scientific runner communicator differs from the native runtime report")
    if route == "MPI_COMM_WORLD":
        return pops.ExecutionContext.mpi_world(artifact)
    if route == "serial":
        return execution_context_for_bind(artifact.platform_manifest, {})
    raise RuntimeError("scientific runner has no authenticated native communicator route")
