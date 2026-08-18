"""PO-07 fail-closed native report."""
from __future__ import annotations

from verification.pops_verify.native_evidence import (
    REDUCED_NOT_SUPPORTED,
    report_from_native_series,
)
from verification.pops_verify.report import write_verification_report

CASE_ID = "PO-07"


def write_po07_report(output_dir, native_series=None) -> dict:
    """Refuse the normative PO-07 ID. Public FFT has no tolerance sweep."""
    return write_verification_report(
        report_from_native_series(
            CASE_ID,
            native_series,
            native_dimensions=[1],
            components=["poisson"],
            variable="potential",
            extra_reasons={"orders": REDUCED_NOT_SUPPORTED[CASE_ID]},
        ),
        output_dir,
    )
