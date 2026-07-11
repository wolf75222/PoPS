import pytest

from pops.codegen.lowering_coverage import LoweringRejection
from pops.fields import FieldProblem, HoldPrevious, Recompute
from pops.ir.expr import Var
from pops.math import laplacian, unknown
from pops.time import always, every


def _problem():
    phi = unknown("phi")
    return FieldProblem(
        name="poisson", unknown=phi,
        equation=(-laplacian(phi) == Var("rho", "cons")), solver=object())


def test_nontrivial_field_cadence_is_never_accepted_as_inert_semantics():
    problem = _problem().solve(every(4), HoldPrevious())
    assert not problem.available()
    with pytest.raises(LoweringRejection, match="ADC-659") as caught:
        problem.validate()
    rejection = caught.value
    assert rejection.gate == "field_cadence_not_lowered"
    row = next(row for row in rejection.report.rows if row.source == rejection.source)
    assert row.disposition == "rejected" and row.targets == ()
    with pytest.raises(LoweringRejection, match="ADC-659"):
        problem.lower()


def test_always_recompute_is_behavior_preserving_and_validates():
    assert _problem().solve(always(), Recompute()).validate() is True


def test_always_hold_is_nontrivial_and_rejected():
    with pytest.raises(LoweringRejection):
        _problem().solve(always(), HoldPrevious()).validate()
