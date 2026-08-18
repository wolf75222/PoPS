"""EU-02 density-error helper and fail-closed native report."""
from __future__ import annotations

from pathlib import Path

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
    """L1/L2/L∞ of numerical density against the translated vortex."""
    x, y, width = _run.cell_centers(n_cells)
    volumes = width * width
    primitives = _run.conserved_to_primitives(conserved)
    oracle = exact_vortex(x, y, t, u_inf=u_inf, v_inf=v_inf)
    return reference_errors(primitives["rho"], oracle["rho"], volumes)


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


def analyze_series(errors, resolutions, output_dir) -> dict:
    """Write a report from a caller-supplied native error series."""
    spacings = [float(_exact.PERIOD) / float(n) for n in resolutions]
    return write_eu02_report(
        output_dir,
        native_series={"linf": list(errors), "spacings": spacings, "variable": "rho"},
    )
