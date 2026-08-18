"""IF-03 in-memory Kokkos Serial vs OpenMP labels, plus optional native.

Each label samples the same exact sine. Optional ``run_native`` reuses
TR-01 at ``OMP_NUM_THREADS=1`` (Serial-like) and ``OMP_NUM_THREADS=8``
(OpenMP). GPU spaces are refused: there is no public CUDA space.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from verification.pops_verify.campaign import resolve_artifact_dim
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")
_TR01_RUN = (
    Path(__file__).resolve().parents[2] / "transport" / "advection_sine" / "run.py"
)


class NativeUnavailable(RuntimeError):
    """Optional native Serial/OpenMP compare cannot run in this environment."""


def exact_fields_for_spaces(n_cells: int = _exact.DEFAULT_N_CELLS, t=0.0):
    """Return exact fields keyed by KokkosSerial and KokkosOpenMP."""
    return {
        name: _exact.exact_on_space(n_cells, name, t)
        for name in _exact.EXECUTION_SPACES
    }


def max_space_difference(n_cells: int = _exact.DEFAULT_N_CELLS, t=0.0) -> float:
    """Return the field-to-field L∞ between Serial and OpenMP labels."""
    fields = exact_fields_for_spaces(n_cells, t)
    volumes = _exact.cell_volumes(n_cells)
    errors = reference_errors(
        fields["KokkosSerial"], fields["KokkosOpenMP"], volumes
    )
    return float(errors.linf)


def space_threads(name: str) -> int:
    """Return the OMP_NUM_THREADS label for a Serial or OpenMP space."""
    if name in _exact.GPU_SPACES:
        raise NativeUnavailable(_exact.CUDA_UNAVAILABLE)
    threads = _exact.SPACE_THREADS.get(name)
    if threads is None:
        raise ValueError(f"unknown execution space {name!r}")
    if name == "KokkosOpenMP":
        return int(os.environ.get("POPS_ORDER_OMP", str(threads)))
    return int(threads)


def _require_dim1(
    artifact_dim: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    resolved = resolve_artifact_dim(cli_value=artifact_dim, environ=environ)
    if resolved is not None and resolved != _exact.REQUIRED_NATIVE_DIM:
        raise NativeUnavailable(
            f"POPS_NATIVE_DIM={resolved!r} does not match required dim "
            f"{_exact.REQUIRED_NATIVE_DIM}; no fallback to another native extension"
        )
    return _exact.REQUIRED_NATIVE_DIM


def _tr01_run():
    return load_sibling_module(_TR01_RUN)


def run_native(
    n_cells: int = _exact.DEFAULT_N_CELLS,
    t_end: float = _exact.DEFAULT_T_END,
    *,
    space: str = "KokkosSerial",
    artifact_dim: int | None = None,
    environ: Mapping[str, str] | None = None,
):
    """Run TR-01 under one Kokkos space label. GPU is refused."""
    _require_dim1(artifact_dim=artifact_dim, environ=environ)
    threads = space_threads(space)
    previous = os.environ.get("OMP_NUM_THREADS")
    os.environ.setdefault("POPS_NATIVE_DIM", str(_exact.REQUIRED_NATIVE_DIM))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        field = _tr01_run().run_native(n_cells, t_end=t_end)
    except NativeUnavailable:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "NativeUnavailable":
            raise NativeUnavailable(str(exc)) from exc
        raise
    finally:
        if previous is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = previous
    return np.reshape(np.asarray(field, dtype=np.float64), (-1,))


def run_native_spaces(
    n_cells: int = _exact.DEFAULT_N_CELLS,
    t_end: float = _exact.DEFAULT_T_END,
    *,
    artifact_dim: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict:
    """Compare TR-01 at OMP_NUM_THREADS=1 vs 8 (Serial vs OpenMP)."""
    _require_dim1(artifact_dim=artifact_dim, environ=environ)
    return {
        name: run_native(
            n_cells,
            t_end,
            space=name,
            artifact_dim=artifact_dim,
            environ=environ,
        )
        for name in _exact.EXECUTION_SPACES
    }


def run_native_gpu(
    n_cells: int = _exact.DEFAULT_N_CELLS,
    t_end: float = _exact.DEFAULT_T_END,
    *,
    space: str = "KokkosCuda",
):
    """Refuse GPU: there is no public CUDA execution space."""
    del n_cells, t_end
    if space not in _exact.GPU_SPACES:
        raise ValueError(f"unknown GPU space {space!r}")
    raise NativeUnavailable(_exact.CUDA_UNAVAILABLE)
