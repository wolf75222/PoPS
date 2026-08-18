"""EU-03 fail-closed native report."""
from __future__ import annotations

from verification.pops_verify.native_evidence import report_from_native_series
from verification.pops_verify.report import write_verification_report

CASE_ID = "EU-03"


def write_eu03_report(output_dir, native_series=None) -> dict:
    """Write artifacts from a native MMS series, or fail closed."""
    return write_verification_report(
        report_from_native_series(
            CASE_ID,
            native_series,
            native_dimensions=[1],
            components=["euler"],
            variable="density",
        ),
        output_dir,
    )
