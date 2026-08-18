"""EU-06 fail-closed native report."""
from __future__ import annotations

from verification.pops_verify.native_evidence import report_from_native_series
from verification.pops_verify.report import write_verification_report

CASE_ID = "EU-06"


def write_eu06_report(output_dir, native_series=None) -> dict:
    """Write artifacts from a native free-stream series, or fail closed."""
    extra = {"orders": "machine-zero free-stream / no spatial-order campaign"}
    return write_verification_report(
        report_from_native_series(
            CASE_ID,
            native_series,
            native_dimensions=[2],
            components=["euler"],
            extra_reasons=extra,
            allow_empty_orders=True,
        ),
        output_dir,
    )
