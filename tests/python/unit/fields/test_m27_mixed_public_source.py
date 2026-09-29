"""The closed periodic mixed field system remains two equations and two unknowns."""
from __future__ import annotations

import pops
import pytest

from pops._ir.elliptic import DivCoeffGrad, Reaction
from pops._ir.quantity import PhysicalSupport
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.fields import (
    CellCenteredGeneralCoupled, CellCenteredSecondOrder, FieldBoundary,
    FieldDiscretization, FieldProblem, FieldProblemError, bcs,
)
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import Handle, OwnerPath
from pops.projection import ConservativeCellAverage
from pops.solvers import CG, GMRES
from pops.time import FailRun, FixedDt


def mixed_case(*, general=True, sampling="cell_average", n=16, permuted=False,
               solver=None):
    frame = CartesianDomain("mixed-periodic", lower=(0.,), upper=(1.,)).frame(Cartesian1D())
    model = pops.Model("concentration", frame=frame)
    state = model.state(
        "c", components=("c",), sampling=sampling,
        support=PhysicalSupport((("x", "mixed-periodic"),)))
    case = pops.Case("mixed-field")
    block = case.block("material", model, states=(state,))
    c, mu = (Handle(name, kind="field", owner=OwnerPath.model("mixed-field"))
             for name in ("c_next", "mu_next"))
    equations = {
        c: Reaction(c, 1) + DivCoeffGrad(mu, 1, scale=-.01) == state[0],
        mu: Reaction(mu, 1) - Reaction(c, 1) + DivCoeffGrad(c, .08**2) == 0,
    }
    unknowns = (mu, c) if permuted else (c, mu)
    problem = FieldProblem(
        "mixed", unknowns=unknowns,
        equations=tuple(equations[unknown] for unknown in unknowns),
        boundaries=tuple(FieldBoundary(
            unknown, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic()))
            for unknown in unknowns))
    method = CellCenteredGeneralCoupled() if general else CellCenteredSecondOrder()
    field = case.field(problem, FieldDiscretization(
        method=method, boundaries=(),
        solver=solver or GMRES(max_iter=200, rel_tol=1e-12, abs_tol=1e-12)))
    program = pops.Program("mixed-be")
    current = program.state(block[state])
    solved = program.solve(field, values={block[state]: current.n},
                           at=current.next.point).consume(action=FailRun())
    observed = field.observe(solved)
    return case, block, state, field, problem, program, current, observed, Uniform(
        CartesianGrid(frame=frame, cells=(n,), periodic=PeriodicAxes(frame.axes)))


@pytest.mark.parametrize("permuted", (False, True))
def test_mixed_c_mu_full_public_lowering_preserves_general_matrix_and_identity(permuted):
    case, block, state, field, problem, program, current, observed, layout = mixed_case(
        permuted=permuted)
    c = next(unknown for unknown in problem.unknowns if unknown.local_id == "c_next")
    mu = next(unknown for unknown in problem.unknowns if unknown.local_id == "mu_next")
    c_next = observed.cell_mean_state(field[c], target=current.next)
    mu_next = observed[field[mu]]
    program.store_history("mu_accepted", mu_next, depth=1)
    program.commit(current.next, c_next)
    program.step_strategy(FixedDt(.01))
    case.program(program)
    case.initials.add(InitialCondition(
        state=block[state], value=BindArray(), projection=ConservativeCellAverage()))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(
        resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "prepare_general_field_coefficients<pops::kNativeDimension, 2, 4, false>" in code
    assert "apply_general_field<pops::kNativeDimension, 2, 4>" in code
    assert "ctx.commit_many(" in code
    assert "mu_accepted" in code


def test_cell_mean_projection_refuses_unstated_sampling_without_program_mutation():
    _case, _block, _state, field, problem, program, current, observed, _layout = mixed_case(
        sampling="unspecified")
    before = (program._next_id, len(program._values))
    with pytest.raises(FieldProblemError, match="cell-volume mean"):
        observed.cell_mean_state(field[problem.unknowns[0]], target=current.next)
    assert (program._next_id, len(program._values)) == before


def test_default_spd_field_route_does_not_admit_general_coupling():
    _case, _block, _state, _field, _problem, program, _current, _observed, _layout = mixed_case(
        general=False)
    coefficient = next(node for node in program._values if node.op == "field_problem_coefficients")
    assert "coefficient_admissibility" not in coefficient.attrs


def test_general_coupling_refuses_cg_spd_claim():
    with pytest.raises((FieldProblemError, ValueError, TypeError), match="CG|SPD|symmetric|positive"):
        mixed_case(solver=CG(max_iter=20))
