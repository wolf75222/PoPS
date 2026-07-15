"""Exact Python projection of the generated PopsExecutionContextV1 resource identity."""
from __future__ import annotations

from typing import Any


def component_execution_data(context: Any) -> dict[str, Any]:
    """Project one installed ExecutionContext without inferring global/default resources."""
    from pops._platform_contracts import ExecutionContext

    if type(context) is not ExecutionContext:
        raise TypeError("native component requires the exact RuntimeInstance ExecutionContext")
    precision_codes = {
        "float16": 1, "bfloat16": 2, "float32": 3, "float64": 4,
    }
    scalar_codes = {"float32": 1, "float64": 2}

    def precision(name: str) -> int:
        value = getattr(context.backend.precision, name).require(
            "ExecutionContext.backend.precision.%s" % name)
        try:
            return precision_codes[value]
        except KeyError:
            raise ValueError("native component ABI v1 cannot represent %s precision %r"
                             % (name, value)) from None

    datatype = context.datatype.identity
    try:
        scalar_type = scalar_codes[datatype]
    except KeyError:
        raise ValueError(
            "native component ABI v1 cannot represent datatype %r" % datatype) from None
    spaces = tuple(context.backend.memory_spaces.require(
        "ExecutionContext.backend.memory_spaces"))
    device = context.device.identity
    if device in ("host", "cpu") and "host" in spaces:
        memory_space = 1
        stream_handle = 0
        stream_identity = "host::synchronous"
    else:
        raise ValueError(
            "native component bridge requires an explicit stream resource for non-host "
            "device %r" % device)
    communicator = context.communicator
    if communicator.identity == "serial":
        if communicator.handle is not None:
            raise ValueError("serial ExecutionContext must not hide a communicator handle")
        if context.datatype.handle is not None:
            raise ValueError("serial ExecutionContext must not hide an MPI datatype handle")
        communicator_f_handle = 0
        communicator_datatype_f_handle = 0
        communicator_datatype_identity = "none"
    elif communicator.identity == "MPI_COMM_WORLD":
        try:
            from mpi4py import MPI
        except ImportError as exc:
            raise RuntimeError(
                "MPI component execution requires mpi4py for the explicit communicator and "
                "datatype handles"
            ) from exc
        if not isinstance(communicator.handle, MPI.Comm) or MPI.Comm.Compare(
                communicator.handle, MPI.COMM_WORLD) != MPI.IDENT:
            raise ValueError(
                "MPI component execution requires the exact mpi4py.MPI.COMM_WORLD handle")
        if context.datatype.handle is not MPI.DOUBLE:
            raise ValueError(
                "MPI component execution requires the exact mpi4py.MPI.DOUBLE datatype handle")
        communicator_f_handle = int(communicator.handle.py2f())
        communicator_datatype_f_handle = int(context.datatype.handle.py2f())
        communicator_datatype_identity = "MPI_DOUBLE"
    else:
        raise TypeError(
            "native component execution supports only serial or exact MPI_COMM_WORLD")
    return {
        "execution_identity": context.identity.token,
        "context_version": 1,
        "memory_space": memory_space,
        "backend_identity": context.backend.identity.token,
        "device_identity": device,
        "scalar_type": scalar_type,
        "storage_precision": precision("storage"),
        "compute_precision": precision("compute"),
        "accumulation_precision": precision("accumulation"),
        "reduction_precision": precision("reduction"),
        "stream_handle": stream_handle,
        "stream_identity": stream_identity,
        "communicator_f_handle": communicator_f_handle,
        "communicator_datatype_f_handle": communicator_datatype_f_handle,
        "communicator_identity": communicator.identity,
        "communicator_datatype_identity": communicator_datatype_identity,
    }


__all__ = ["component_execution_data"]
