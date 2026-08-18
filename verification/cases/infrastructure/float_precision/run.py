"""IF-09 in-memory float32 / float64 evaluation.

Evaluate the TR-01 sine in both storage dtypes and report L∞(f32, f64).
Public float32 Case authoring is not active
(``pops.runtime_environment.supports_single_precision`` is false;
``validate_precision("float32")`` refuses). ``run_native`` raises
``NativeUnavailable`` with that missing capability.
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_v15 = load_sibling_module(Path(__file__).resolve().parents[1] / "_v15.py")

FLOAT32_AUTHORING_REFUSAL = "public float32 Case authoring not active"


class NativeUnavailable(RuntimeError):
    """Raised when a public float32 Case cannot be authored."""


def evaluate_precisions(n_cells: int = _exact.DEFAULT_N_CELLS, t=_exact.T):
    """Return TR-01 sine in float32 and float64 plus the cell volumes."""
    centers = _exact.cell_centers(n_cells)
    return {
        "float32": _exact.exact_sine_as(centers, t, _exact.DTYPES[0]),
        "float64": _exact.exact_sine_as(centers, t, _exact.DTYPES[1]),
        "volumes": _exact.cell_volumes(n_cells),
        "centers": centers,
    }


def max_precision_difference(n_cells: int = _exact.DEFAULT_N_CELLS, t=_exact.T) -> float:
    """Return L∞ between the float32 and float64 TR-01 fields."""
    fields = evaluate_precisions(n_cells, t)
    errors = reference_errors(fields["float32"], fields["float64"], fields["volumes"])
    return float(errors.linf)


def refuse_float32_case_authoring() -> str:
    """Return why a float32 native Case is not authored."""
    return FLOAT32_AUTHORING_REFUSAL


def run_native(n_cells: int = _exact.DEFAULT_N_CELLS, t_end=_exact.T, request=None):
    """Refuse a native float32 Case. Keep the in-memory f32/f64 sine."""
    from pops.runtime_environment import RuntimeCapabilityError, validate_precision

    _v15.bind_campaign(request, NativeUnavailable)
    del n_cells, t_end
    try:
        validate_precision("float32", where="IF-09")
    except RuntimeCapabilityError as exc:
        raise NativeUnavailable(refuse_float32_case_authoring()) from exc
    raise NativeUnavailable(refuse_float32_case_authoring())
