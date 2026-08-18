"""Shared IF campaign bind: fail-closed modes, truthful provenance, artifacts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ALLOWED_MPI_MODES = ("off", "on")
ALLOWED_SPACES = ("KokkosSerial", "KokkosOpenMP", "KokkosCuda")
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
MAX_NODES = 2


class ModeRefusal(ValueError):
    """Requested MPI or execution-space mode cannot be bound."""


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
