"""TM-01 fail-closed native temporal report."""
from __future__ import annotations

from verification.pops_verify.native_evidence import report_from_native_series
from verification.pops_verify.report import write_verification_report

CASE_ID = "TM-01"
ORDER_THRESHOLD = 1.8


def write_tm01_report(output_dir, native_series=None, **_unused) -> dict:
    """Write artifacts from a native Δt series, or fail closed."""
    return write_verification_report(
        report_from_native_series(
            CASE_ID,
            native_series,
            native_dimensions=[1],
            components=["time"],
            kind="temporal",
            variable="q",
            threshold=ORDER_THRESHOLD,
        ),
        output_dir,
    )


