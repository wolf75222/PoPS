"""EU-02 density-error helper and fail-closed native report."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.native_evidence import report_from_native_series
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import write_verification_report

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")
_run = load_sibling_module(_CASE_DIR / "run.py")
exact_vortex = _exact.exact_vortex
CASE_ID = "EU-02"
ORDER_THRESHOLD = 1.8


def density_error(conserved, n_cells, t, *, u_inf=1.0, v_inf=0.0):
    """L1/L2/L∞ of numerical density against cell-average vortex density."""
    from verification.pops_verify.cell_averages import analytic_cell_averages

    x, y, width = _run.cell_centers(n_cells)
    volumes = width * width
    primitives = _run.conserved_to_primitives(conserved)
    axis_lo = np.arange(int(n_cells), dtype=np.float64) * width
    axis_hi = axis_lo + width
    x_lo, y_lo = np.meshgrid(axis_lo, axis_lo, indexing="xy")
    x_hi, y_hi = np.meshgrid(axis_hi, axis_hi, indexing="xy")
    lo = np.stack((x_lo, y_lo), axis=-1)
    hi = np.stack((x_hi, y_hi), axis=-1)

    def density(xx, yy):
        return exact_vortex(xx, yy, t, u_inf=u_inf, v_inf=v_inf)["rho"]

    oracle = analytic_cell_averages(density, lo, hi)
    return reference_errors(primitives["rho"], oracle, volumes)


def write_eu02_report(output_dir, native_series=None) -> dict:
    """Write artifacts from a native density-error series, or fail closed."""
    return write_verification_report(
        report_from_native_series(
            CASE_ID,
            native_series,
            native_dimensions=[2],
            components=["euler"],
            variable="rho",
            threshold=ORDER_THRESHOLD,
        ),
        output_dir,
    )


