"""TR-02 mass/barycenter utilities and fail-closed native report."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from verification.pops_verify.native_evidence import report_from_native_series
from verification.pops_verify.report import write_verification_report

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CASE_ID = "TR-02"


def pulse_mass(q, volumes) -> float:
    """Return the discrete mass sum_i q_i V_i."""
    field = np.asarray(q, dtype=np.float64)
    cell_volumes = np.asarray(volumes, dtype=np.float64)
    return float(np.sum(field * cell_volumes))


def pulse_barycenter(x, q, volumes, q0) -> float:
    """Periodic first moment of q-q0 on [0, 1], unwrapped about the peak."""
    centers = np.asarray(x, dtype=np.float64)
    weights = (np.asarray(q, dtype=np.float64) - float(q0)) * np.asarray(
        volumes, dtype=np.float64
    )
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("non-positive pulse mass")
    peak = float(centers[int(np.argmax(weights))])
    unwrap = np.mod(centers - peak + 0.5, 1.0) - 0.5
    return float(np.mod(peak + float(np.sum(unwrap * weights)) / total, 1.0))


def write_tr02_report(output_dir, native_series=None) -> dict:
    """Write artifacts from a native error series, or fail closed without one."""
    return write_verification_report(
        report_from_native_series(
            CASE_ID,
            native_series,
            native_dimensions=[1],
            components=["transport"],
        ),
        output_dir,
    )
