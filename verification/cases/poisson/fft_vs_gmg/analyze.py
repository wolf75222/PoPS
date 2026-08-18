"""PO-05 fail-closed native report."""
from __future__ import annotations

from verification.pops_verify.native_evidence import (
    REDUCED_NOT_SUPPORTED,
    report_from_native_series,
)
from verification.pops_verify.report import write_verification_report

CASE_ID = "PO-05"


def write_po05_report(output_dir, native_series=None) -> dict:
    """Refuse the normative PO-05 ID. Uniform FFT/CG is not FFT-vs-GMG."""
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
