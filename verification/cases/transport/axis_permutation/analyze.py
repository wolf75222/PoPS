"""TR-06 fail-closed native report. Exact permutation identities stay in run.py."""
from __future__ import annotations

from verification.pops_verify.native_evidence import report_from_native_series
from verification.pops_verify.report import write_verification_report

CASE_ID = "TR-06"


def write_tr06_report(output_dir, native_series=None) -> dict:
    """Write artifacts from a native permutation series, or fail closed."""
    return write_verification_report(
        report_from_native_series(
            CASE_ID,
            native_series,
            native_dimensions=[2],
            components=["transport"],
            extra_reasons={
                "orders": (
                    "axis-permutation / reflection identity; no spatial-order "
                    "campaign"
                )
            },
            allow_empty_orders=True,
        ),
        output_dir,
    )
