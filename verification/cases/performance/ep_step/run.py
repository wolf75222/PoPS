"""PF-06 in-memory Euler–Poisson step segmentation plus optional CP-02 timing.

Records a fake timing for each pipeline stage. The total is the sum.

``run_native`` times CP-02 ``langmuir_cold.run_native`` and returns elapsed
plus cells/s. GPU spaces are refused. ``pops.run`` stays inside CP-02.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_CP02_RUN = (
    Path(__file__).resolve().parents[2]
    / "euler_poisson"
    / "langmuir_cold"
    / "run.py"
)

DEFAULT_NATIVE_N_CELLS = 32
DEFAULT_NATIVE_T_END = 0.05
GPU_SPACES = ("KokkosCuda", "KokkosHIP")
CUDA_UNAVAILABLE = "no public CUDA space"


class NativeUnavailable(RuntimeError):
    """Optional native CP-02 timing cannot run in this environment."""


def run_ep_step() -> dict:
    """Run the toy EP pipeline and return fake segment timings."""
    timings = _exact.segment_timings()
    return {
        "segments": list(_exact.SEGMENT_NAMES),
        "timings": timings,
    }


def _cp02_run():
    """Load CP-02 ``run.py`` via ``load_sibling_module``."""
    return load_sibling_module(_CP02_RUN)


def _reraise_native_unavailable(exc: BaseException) -> None:
    if exc.__class__.__name__ == "NativeUnavailable":
        raise NativeUnavailable(str(exc)) from exc
    raise exc


def _cells_per_second(n_cells: int, elapsed_s: float):
    if elapsed_s <= 0.0:
        return None
    return float(n_cells) / float(elapsed_s)



def run_native(*args, **kwargs):
    """PF timed work belongs to benchmarks/manifest.toml, not a sibling wrap."""
    from verification.pops_verify.official_benchmark import refuse_unofficial_pf

    raise NativeUnavailable(refuse_unofficial_pf('PF-06'))

