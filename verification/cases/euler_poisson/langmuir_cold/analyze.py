"""CP-02 analysis from native fields; fail-closed without Kokkos output."""
from __future__ import annotations

import sys
from pathlib import Path

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.native_evidence import (
    fail_closed_report,
    native_diagnostics,
    native_report_sections,
    order_rows,
)
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")
CASE_ID = "CP-02"
NATIVE_DIMS = [1]
SUITE = "pr"
COMPONENT = "euler_poisson"
BLOCKER = (
    "no native Kokkos output; exact-vs-exact and synthetic identities "
    "are not scientific evidence"
)


def analyze_native(native):
    """Compute phase/frequency/field/energy/order from native arrays only."""
    return native_diagnostics(native)


def write_cp02_report(output_dir, *, native=None, request=None) -> dict:
    """Write artifacts. Required cases fail; never not-supported unless gated."""
    orders = []
    poisson = None
    coupling = None
    extra_reasons = None
    reason = BLOCKER
    if native is not None:
        diagnostics = analyze_native(native)
        orders = order_rows(CASE_ID, diagnostics)
        sections = native_report_sections(diagnostics)
        poisson = sections["poisson"]
        coupling = sections["coupling"]
        extra_reasons = sections["extra_reasons"]
        reason = (
            "native diagnostics computed from supplied arrays; Kokkos campaign "
            "is not authenticated in this isolated stream"
        )
    return write_verification_report(
        fail_closed_report(
            case_id=CASE_ID,
            component=COMPONENT,
            native_dims=list(NATIVE_DIMS),
            reason=reason,
            suite=SUITE,
            request=request,
            orders=orders,
            poisson=poisson,
            coupling=coupling,
            extra_reasons=extra_reasons,
        ),
        output_dir,
    )
