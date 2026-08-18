"""TR-07 fail-closed native report. TV/overshoot utilities stay in run.py."""
from __future__ import annotations

from verification.pops_verify.native_evidence import report_from_native_series
from verification.pops_verify.report import write_verification_report

CASE_ID = "TR-07"


def write_tr07_report(output_dir, native_series=None) -> dict:
    """Write artifacts from a native slot field, or fail closed."""
    return write_verification_report(
        report_from_native_series(
            CASE_ID,
            native_series,
            native_dimensions=[1],
            components=["transport"],
            extra_reasons={"orders": "discontinuous / limiter, not order-2"},
            allow_empty_orders=True,
        ),
        output_dir,
    )
