"""IF-02 in-memory OpenMP-style thread labels plus optional native sweep.

Each label samples the same exact sine on a different static thread count.
``run_native_threads`` sets ``OMP_NUM_THREADS`` and reuses TR-01 ``run_native``.
"""
from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.native_evidence import campaign_run_fields
from verification.pops_verify.reference_errors import reference_errors

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_v15 = load_sibling_module(Path(__file__).resolve().parents[1] / "_v15.py")
_TR01_RUN = (
    Path(__file__).resolve().parents[2] / "transport" / "advection_sine" / "run.py"
)


class NativeUnavailable(RuntimeError):
    """Raised when the optional native thread sweep cannot run."""


def thread_counts():
    """Return the canonical 1 / 2 / 4 / 8 thread labels."""
    return _exact.THREAD_COUNTS


def exact_fields_for_thread_counts(n_cells: int = _exact.DEFAULT_N_CELLS, t=0.0):
    """Return exact fields keyed by the four OpenMP thread-count labels."""
    return {
        n_threads: _exact.exact_on_thread_count(n_cells, n_threads, t)
        for n_threads in _exact.THREAD_COUNTS
    }


def max_thread_difference(n_cells: int = _exact.DEFAULT_N_CELLS, t=0.0) -> float:
    """Return the max pairwise L∞ between the four exact thread labels."""
    fields = exact_fields_for_thread_counts(n_cells, t)
    volumes = _exact.cell_volumes(n_cells)
    linf = 0.0
    for left, right in combinations(fields.values(), 2):
        errors = reference_errors(left, right, volumes)
        linf = max(linf, errors.linf)
    return float(linf)


def _tr01_run():
    """Load TR-01 ``run.py`` via ``load_sibling_module``."""
    return load_sibling_module(_TR01_RUN)


def _restore_env(name: str, previous) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _reraise_native_unavailable(exc: BaseException) -> None:
    if exc.__class__.__name__ == "NativeUnavailable":
        raise NativeUnavailable(str(exc)) from exc
    raise exc


def run_native_threads(
    thread_counts=(1, 2, 4, 8),
    n_cells: int = 32,
    t_end: float = 0.25,
):
    """Set ``OMP_NUM_THREADS``, call TR-01 ``run_native``, return pairwise L∞."""
    counts = tuple(int(n_threads) for n_threads in thread_counts)
    if not counts:
        raise ValueError("thread_counts must be non-empty")
    tr01 = _tr01_run()
    fields = {}
    previous = os.environ.get("OMP_NUM_THREADS")
    try:
        for n_threads in counts:
            os.environ["OMP_NUM_THREADS"] = str(n_threads)
            try:
                field = np.asarray(tr01.run_native(n_cells, t_end=t_end), dtype=np.float64)
            except Exception as exc:
                _reraise_native_unavailable(exc)
            fields[n_threads] = np.ravel(field)
    finally:
        _restore_env("OMP_NUM_THREADS", previous)
    volumes = _exact.cell_volumes(n_cells)
    pairwise = {}
    for left, right in combinations(counts, 2):
        errors = reference_errors(fields[left], fields[right], volumes)
        pairwise[(left, right)] = float(errors.linf)
    return pairwise


def run_native(n_cells: int = 32, t_end: float = 0.25, request=None):
    """Bind an authenticated Kokkos OpenMP leaf and compare native thread counts.

    Environment-only ``OMP_NUM_THREADS`` on a Serial leaf is not OpenMP proof.
    """
    _v15.bind_campaign(request, NativeUnavailable)
    if request is not None and request.min_resolution is not None:
        n_cells = int(request.min_resolution)
    if request is None or request.execution_space != "KokkosOpenMP":
        raise NativeUnavailable(
            "IF-02 requires CampaignRequest with execution_space=KokkosOpenMP; "
            "OMP_NUM_THREADS alone is not OpenMP proof"
        )
    try:
        backend = _v15.require_kokkos_openmp()
    except RuntimeError as exc:
        raise NativeUnavailable(str(exc)) from exc
    requested = int(getattr(request.resources, "omp_threads", None) or 4)
    counts = tuple(sorted({1, max(1, requested)}))
    if len(counts) == 1:
        counts = (1, max(2, requested))
    fields = {}
    previous = os.environ.get("OMP_NUM_THREADS")
    try:
        for n_threads in counts:
            os.environ["OMP_NUM_THREADS"] = str(n_threads)
            try:
                import pops

                if hasattr(pops, "set_threads"):
                    pops.set_threads(n_threads)
            except Exception:
                pass
            try:
                field = np.asarray(
                    _tr01_run().run_native(n_cells, t_end=t_end), dtype=np.float64
                )
            except Exception as exc:
                _reraise_native_unavailable(exc)
                raise NativeUnavailable(str(exc)) from exc
            fields[n_threads] = np.ravel(field)
    finally:
        _restore_env("OMP_NUM_THREADS", previous)
    volumes = _exact.cell_volumes(n_cells)
    pairwise = {}
    for left, right in combinations(counts, 2):
        errors = reference_errors(fields[left], fields[right], volumes)
        pairwise[f"{left}-{right}"] = float(errors.linf)
    return campaign_run_fields(
        request=request,
        n_cells=n_cells,
        t_end=t_end,
        time_program="SSPRK2",
        cfl=0.45,
        comparison={
            "kind": "openmp_threads",
            "backend": backend,
            "threads": list(counts),
            "pairwise_linf": pairwise,
        },
        kokkos_execution_space="KokkosOpenMP",
    )
