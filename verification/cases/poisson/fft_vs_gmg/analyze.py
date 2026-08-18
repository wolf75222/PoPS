"""PO-05 fail-closed native report."""
from __future__ import annotations

from verification.pops_verify.native_evidence import report_from_native_series
from verification.pops_verify.report import write_verification_report

CASE_ID = "PO-05"


def write_po05_report(output_dir, native_series=None) -> dict:
    """Write artifacts from a native FFT/GMG pair, or fail closed."""
    return write_verification_report(
        report_from_native_series(
            CASE_ID,
            native_series,
            native_dimensions=[1],
            components=["poisson"],
            variable="potential",
            extra_reasons={
                "orders": (
                    "uniform GeometricMG lowers to CartesianCG; not a "
                    "composite AMR MG comparison"
                )
            },
            allow_empty_orders=True,
        ),
        output_dir,
    )
