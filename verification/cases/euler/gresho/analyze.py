"""EU-05 fail-closed native report."""
from __future__ import annotations

from verification.pops_verify.native_evidence import report_from_native_series
from verification.pops_verify.report import write_verification_report

CASE_ID = "EU-05"


def write_eu05_report(output_dir, native_series=None) -> dict:
    """Write artifacts from a native Gresho series, or fail closed."""
    extra = {"orders": "stationary vortex / no spatial-order campaign"}
    return write_verification_report(
        report_from_native_series(
            CASE_ID,
            native_series,
            native_dimensions=[2],
            components=["euler"],
            variable="rho",
            extra_reasons=extra,
            allow_empty_orders=True,
        ),
        output_dir,
    )
