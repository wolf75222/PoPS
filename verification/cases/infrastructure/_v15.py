"""Shared IF campaign bind: fail-closed modes, native world, truthful provenance."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ALLOWED_MPI_MODES = ("off", "on")
ALLOWED_SPACES = ("KokkosSerial", "KokkosOpenMP", "KokkosCuda")
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
MAX_NODES = 2
INJECTED_NATIVE_ENV = "POPS_VERIFY_NATIVE_EXE"


class ModeRefusal(ValueError):
    """Requested MPI or execution-space mode cannot be bound."""


class NativeWorldError(RuntimeError):
    """Raised when the authenticated native communicator is missing or too small."""


def injected_missing_native() -> str | None:
    """Return a missing-binary reason when tests inject a nonexistent path."""
    raw = os.environ.get(INJECTED_NATIVE_ENV)
    if raw is None or str(raw).strip() == "":
        return None
    path = Path(raw)
    if path.is_file():
        return None
    return f"injected native executable missing: {path}"


def refuse_missing_injected_native() -> None:
    """Raise RuntimeError when an injected native path is absent."""
    reason = injected_missing_native()
    if reason:
        raise RuntimeError(reason)


def bind_campaign(request, unavailable_cls: type[Exception]) -> None:
    """Refuse an injected missing binary, then an invalid campaign mode."""
    reason = injected_missing_native()
    if reason:
        raise unavailable_cls(reason)
    refuse_invalid_mode(request)


def refuse_invalid_mode(request) -> None:
    """Refuse an invalid mpi_mode, execution space, or two-node overflow first."""
    if request is None:
        return
    mode = getattr(request, "mpi_mode", "off")
    if mode not in ALLOWED_MPI_MODES:
        raise ModeRefusal(f"invalid mpi mode {mode!r}")
    space = getattr(request, "execution_space", "KokkosSerial")
    if space not in ALLOWED_SPACES:
        raise ModeRefusal(f"invalid execution space {space!r}")
    nodes = int(getattr(getattr(request, "resources", None), "nodes", 1) or 1)
    if nodes > MAX_NODES:
        raise ModeRefusal(f"requested_nodes {nodes} exceeds two-node limit")


def _selected_module() -> Any:
    try:
        from verification.pops_verify.mpi_world import native_world_size as _unused

        del _unused
    except Exception:
        pass
    from pops._native_selector import selected_native_module

    return selected_native_module(required=False)


def native_world_size(*, required: bool = False) -> int | None:
    """Authenticated native communicator size. Never launcher env."""
    try:
        from verification.pops_verify.mpi_world import NativeWorldError as SharedError
        from verification.pops_verify.mpi_world import native_world_size as shared_size

        try:
            return shared_size(required=required)
        except SharedError as exc:
            if required:
                raise NativeWorldError(str(exc)) from exc
            return None
    except ImportError:
        pass
    try:
        module = _selected_module()
        if module is None:
            raise NativeWorldError("native communicator unavailable")
        n_ranks = getattr(module, "n_ranks", None)
        if not callable(n_ranks):
            raise NativeWorldError("native n_ranks unavailable")
        return int(n_ranks())
    except NativeWorldError:
        if required:
            raise
        return None
    except Exception as exc:
        if required:
            raise NativeWorldError(f"native communicator unavailable: {exc}") from exc
        return None


def native_execution_backend() -> str | None:
    """Return the authenticated Kokkos DefaultExecutionSpace name, if loaded."""
    try:
        module = _selected_module()
    except Exception:
        return None
    if module is None:
        return None
    resource_fn = getattr(module, "native_execution_resource", None)
    if not callable(resource_fn):
        return None
    try:
        resource = resource_fn()
    except Exception:
        return None
    backend = getattr(resource, "execution_backend", None)
    if backend is None:
        return None
    return str(backend)


def require_kokkos_openmp() -> str:
    """Refuse Serial or missing backends when OpenMP is required."""
    backend = native_execution_backend()
    if backend is None:
        raise RuntimeError("Kokkos OpenMP backend unavailable")
    lowered = backend.lower()
    if "openmp" not in lowered or "serial" == lowered:
        raise RuntimeError(
            f"Kokkos OpenMP required; authenticated backend is {backend!r} (Serial leaf cannot pass)"
        )
    return backend


def native_has_parallel_hdf5() -> bool:
    try:
        module = _selected_module()
    except Exception:
        return False
    if module is None:
        return False
    return getattr(module, "__has_parallel_hdf5__", False) is True


def campaign_run_fields(
    request,
    *,
    n_cells: int,
    t_end: float,
    comparison: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Return RUN_FIELDS plus comparison artifacts. Does not invent MPI ranks."""
    mpi_on = request is not None and getattr(request, "mpi_mode", "off") == "on"
    space = getattr(request, "execution_space", None) or "KokkosSerial"
    resources = getattr(request, "resources", None)
    ranks = int(getattr(resources, "mpi_ranks", None) or 1)
    threads = int(getattr(resources, "omp_threads", None) or 1)
    count = int(n_cells)
    fields: dict[str, Any] = {
        "compiler": os.environ.get("CXX", "c++"),
        "build_type": "native-dsl",
        "precision": "float64",
        "kokkos_execution_space": space,
        "mpi_enabled": mpi_on,
        "mpi_library": "none" if not mpi_on else (os.environ.get("POPS_MPI_LIBRARY") or "unknown"),
        "mpi_thread_level_requested": "none" if not mpi_on else "MPI_THREAD_SINGLE",
        "mpi_thread_level_provided": "none" if not mpi_on else "MPI_THREAD_SINGLE",
        "hdf5_collective_enabled": False,
        "mpi_ranks": ranks if mpi_on else 1,
        "omp_threads_per_rank": threads,
        "gpus": 0,
        "resolution": [count],
        "block_size": [count],
        "amr_total_levels": 1,
        "refinement_ratio": 2,
        "subcycling": False,
        "time_program": "SSPRK2",
        "cfl": 0.45,
        "final_time": float(t_end),
        "comparison_artifacts": comparison or {"kind": "none", "paths": []},
    }
    fields.update(overrides)
    return fields


def is_hdf5_file(path: Path) -> bool:
    """Return True when the file starts with the HDF5 signature."""
    try:
        with open(path, "rb") as handle:
            return handle.read(8) == HDF5_MAGIC
    except OSError:
        return False


def refuse_npz_as_hdf5(path: Path) -> None:
    """NPZ or any non-HDF5 blob is not an IF-10 reread."""
    if path.suffix == ".npz" or not is_hdf5_file(path):
        raise RuntimeError(f"{path} is not HDF5; NPZ is not an HDF5 stand-in")


def authenticated_hdf5_collective(path: Path | None = None) -> bool:
    """Collective HDF5 is native parallel-HDF5 capability plus an actual HDF5 path."""
    if not native_has_parallel_hdf5():
        return False
    if path is None:
        return False
    return is_hdf5_file(Path(path))
