"""PO-04 fail-closed native report."""
from __future__ import annotations

from verification.pops_verify.native_evidence import report_from_native_series
from verification.pops_verify.report import write_verification_report

CASE_ID = "PO-04"


def write_po04_report(output_dir, native_series=None) -> dict:
    """Write artifacts from a native Huang–Greengard series, or fail closed."""
    extra = {
        "amr.*": "public composite AMR Poisson is not supported for PO-04",
    }
    return write_verification_report(
        report_from_native_series(
            CASE_ID,
            native_series,
            native_dimensions=[1],
            components=["poisson"],
            variable="potential",
            extra_reasons=extra,
        ),
        output_dir,
    )
