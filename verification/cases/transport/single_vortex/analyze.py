"""TR-03 fail-closed native report. Oracle utilities stay on exact.py / run.py."""
from __future__ import annotations

from verification.pops_verify.native_evidence import report_from_native_series
from verification.pops_verify.report import write_verification_report

CASE_ID = "TR-03"


def write_tr03_report(output_dir, native_series=None) -> dict:
    """Write artifacts from a native return-error series, or fail closed."""
    return write_verification_report(
        report_from_native_series(
            CASE_ID,
            native_series,
            native_dimensions=[2],
            components=["transport"],
            extra_reasons={
                "orders": "return-to-IC / no spatial-order campaign"
                if native_series is not None
                else "no native result series",
            },
            allow_empty_orders=True,
        ),
        output_dir,
    )
