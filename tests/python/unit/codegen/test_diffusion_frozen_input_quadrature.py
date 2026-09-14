"""An independent solved field permits only the explicitly proved Euler update."""
from fractions import Fraction
from types import SimpleNamespace
import pytest

from pops.codegen.program_diffusion_exchanges import (
    _single_forward_euler_with_frozen_inputs, accepted_diffusive_quadrature,
)
from tests.python.integration.runtime.test_public_drift_diffusion_matrix import _solved_potential_case


def _clone(value, **changes):
    fields = {name:getattr(value,name) for name in ("op","attrs","inputs","block","point","state_ref")}
    return SimpleNamespace(**{**fields,**changes})


def test_exact_euler_face_quadrature_survives_an_independent_unchanged_driver():
    case, _ = _solved_potential_case(16,1e-4)
    program = case._time
    rows = accepted_diffusive_quadrature(program)
    assert len(rows) == 1 and rows[0][1] == {1:Fraction(1)}
    assert _single_forward_euler_with_frozen_inputs(program,rows)
    assert any(value.op == "solve_outcome" for value in program._values)


@pytest.mark.parametrize("fault",("changed_driver","wrong_coefficient","wrong_seed","extra_rhs"))
def test_frozen_input_euler_proof_refuses_different_accepted_equations(fault):
    case, _ = _solved_potential_case(16,1e-4)
    program = case._time
    rows = accepted_diffusive_quadrature(program)
    rate = rows[0][0]
    commits = dict(program._commits)
    selected = next(key for key,value in commits.items() if any(item is rate for item in value.inputs))
    driver = next(key for key in commits if key != selected)
    end = commits[selected]
    if fault == "changed_driver":
        value = commits[driver]
        commits[driver] = _clone(value, attrs={**value.attrs,"coeffs":({0:2},)})
    elif fault == "wrong_coefficient":
        commits[selected] = _clone(end, attrs={**end.attrs,"coeffs":({0:1},{1:Fraction(1,2)})})
    elif fault == "wrong_seed":
        from pops.time.points import TimePoint
        seed = _clone(rate.inputs[0],point=TimePoint(program.clock,step=1))
        replacement = _clone(rate,inputs=(seed,*rate.inputs[1:]))
        commits[selected] = _clone(end,inputs=(end.inputs[0],replacement))
        rows = ((replacement,{1:Fraction(1)}),)
    else:
        extra = _clone(rate,op="rhs")
        commits[selected] = _clone(end,inputs=(*end.inputs,extra),
            attrs={**end.attrs,"coeffs":(*end.attrs["coeffs"],{1:1})})
    assert not _single_forward_euler_with_frozen_inputs(SimpleNamespace(_commits=commits),rows)
