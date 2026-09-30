"""Original coupled spatial residuals, exact captures and untouched legacy routes."""
from dataclasses import replace

import pops
import pytest

from pops._ir.elliptic import DivCoeffGrad, Reaction
from pops._ir.handle_expr import ValueExpr
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.fields import (CellCenteredNonlinearCoupled, FieldBoundary, FieldDiscretization,
                         FieldProblem, FieldProblemError, bcs)
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import Handle, OwnerPath
from pops.projection import ConservativeCellAverage
from pops.solvers import Newton, GMRES
from pops.time import FailRun, FixedDt, SolveRequestError


def mixed_case(order=(0, 1), *, width=2, solver=None, boundary=None):
    frame = CartesianDomain("nonlinear-box", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("forcing", frame=frame)
    forcing = model.state("load", components=tuple("f%d" % i for i in range(width)))
    case = pops.Case("generic-original-residual")
    block = case.block("physical-load", model, states=(forcing,))
    unknowns = tuple(Handle("unknown_%d" % i, kind="field", owner=OwnerPath.model("spatial-product"))
                     for i in range(width))
    q, v = unknowns[:2]
    equations = [Reaction(q, 1 + ValueExpr(q)**2) - DivCoeffGrad(v, 1) == forcing[0],
                 Reaction(v, 1) + Reaction(q, ValueExpr(q)) + DivCoeffGrad(q, .1) == forcing[1]]
    for index in range(2, width):
        z = unknowns[index]
        equations.append(Reaction(z, 1 + ValueExpr(z)**2) - DivCoeffGrad(z, .3) == forcing[index])
    selected = tuple(unknowns[index] for index in order)
    problem = FieldProblem("generic-nonlinear", unknowns=selected,
        equations=tuple(equations[index] for index in order), boundaries=tuple(FieldBoundary(
            unknown, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), boundary or bcs.Periodic()))
            for unknown in selected))
    field = case.field(problem, FieldDiscretization(method=CellCenteredNonlinearCoupled(
        finite_difference_step=1e-7), boundaries=(), solver=solver or Newton(tolerance=1e-11)))
    program = pops.Program("spatial-product")
    current = program.state(block[forcing])
    request = field.bind_program_inputs(program=program, values={block[forcing]: current.n},
                                        at=current.next.point, solver=solver or field.default_program_solver())
    return case, field, program, current, request, block, forcing, frame


def finish(case, field, program, current, request, block, forcing, frame, *, mutate=None):
    solved = program.solve(request, solver=field.default_program_solver()).consume(action=FailRun())
    observed = field.observe(solved)
    for index, handle in enumerate(field._field_registry.resolved_registration(field).operator.unknowns):
        program.store_history("observed_%d" % index, observed[handle], depth=1)
    program.commit(current.next, program.value("constant-load", 1 * current.n, at=current.next.point))
    program.step_strategy(FixedDt(.01))
    if mutate is not None:
        mutate(program)
    case.program(program)
    case.initials.add(InitialCondition(state=block[forcing], value=BindArray(), projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 6), periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    return emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))


@pytest.mark.parametrize("width,order", [(2, (0,1)), (2, (1,0)), (3, (2,0,1))])
def test_full_public_original_residual_codegen(width, order):
    args = mixed_case(order, width=width)
    case, field, program, current, request, block, forcing, frame = args
    code = finish(case, field, program, current, request, block, forcing, frame)
    token = next(value for value in program._values if value.op == "solve_spatial_field")
    assert token.attrs["contract"] == "pops.spatial-field-residual@1"
    assert token.attrs["ncomp"] == width
    assert token.attrs["solve_request"]["derivative"]["scheme"] == "central_full_residual"
    assert "PreparedSpatialResidual" in code and "apply_general_field<pops::kNativeDimension, %d, %d>" % (width,width*width) in code
    assert "original_field_residual_recheck_failed" in code
    assert "pops::for_each_cell" in code and "nonfinite_original_field_residual" in code
    assert "block_inverse" not in code and "condensed" not in code


def test_seed_is_initialization_not_equation_capture():
    _, field, program, _, request, *_ = mixed_case()
    first = program.solve(request, solver=field.default_program_solver())
    seed = program.scalar_field("different-seed", ncomp=2)
    second = program.solve(replace(request, seeds={"field_tuple": seed}), solver=field.default_program_solver())
    assert first._token.attrs["solve_request"]["equation_identity"] == second._token.attrs["solve_request"]["equation_identity"]
    assert first._token.attrs["solve_request"]["initialization_identity"] != second._token.attrs["solve_request"]["initialization_identity"]


def test_source_equation_drift_refused():
    _, field, program, _, request, *_ = mixed_case()
    outcome = program.solve(request, solver=field.default_program_solver())
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
    token = outcome._token
    forged = program._replace_value(token, attrs={**token.attrs, "local_expressions": ()})
    with pytest.raises(SolveRequestError, match="equation_identity_drift"):
        validate_nonlinear_field_request(program, forged)


def test_resealed_equation_forgery_still_refused_against_registered_physics():
    from pops.fields._program_nonlinear_problem import _request_data, validate_nonlinear_field_request
    from pops.time.solve_request import SolveUnknown
    from pops.identity.scalar import scalar_literal
    from pops.time._program.serialization import _json_ready

    def corrupt(program):
        token = next(node for node in program._values if node.op == "solve_spatial_field")
        changed = (("add", token.attrs["local_expressions"][0],
                    ("literal", scalar_literal(1).to_data())), *token.attrs["local_expressions"][1:])
        source = {**token.attrs["source_contract"], "local_expressions": changed}
        token = program._replace_value(token, attrs={**token.attrs,
            "local_expressions": changed, "source_contract": source})
        request = token.attrs["solve_request"]
        updated = _request_data(program, token, SolveUnknown("field_tuple", token.inputs[0]),
                                _json_ready(request["physical_problem"]))
        token = program._replace_value(token, attrs={**token.attrs, "solve_request": updated})
        validate_nonlinear_field_request(program, token)  # internally self-consistent

    with pytest.raises(ValueError, match="registered equations"):
        finish(*mixed_case(), mutate=corrupt)


@pytest.mark.parametrize("step", (0, -1, float("nan"), float("inf")))
def test_invalid_fd_selection_refused(step):
    with pytest.raises((TypeError, ValueError)):
        CellCenteredNonlinearCoupled(finite_difference_step=step)


def test_wrong_solver_refused_before_authoring_mutation():
    with pytest.raises(FieldProblemError, match="Newton/GMRES"):
        mixed_case(solver=GMRES(max_iter=20))


def test_nonhomogeneous_boundary_refused():
    with pytest.raises(FieldProblemError, match="periodic or homogeneous Neumann"):
        mixed_case(boundary=bcs.Neumann(1))


@pytest.mark.parametrize("bad", ({"capture_count": 0}, {"seed_index": 99}))
def test_input_slot_forgery_refused(bad):
    _, field, program, _, request, *_ = mixed_case()
    token = program.solve(request, solver=field.default_program_solver())._token
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
    forged = program._replace_value(token, attrs={**token.attrs, **bad})
    with pytest.raises(SolveRequestError, match="input slots"):
        validate_nonlinear_field_request(program, forged)


def test_wrong_seed_width_refuses_atomically():
    _, field, program, _, request, *_ = mixed_case()
    seed = program.scalar_field("foreign-width", ncomp=3)
    before = (program._next_id, tuple(program._values))
    with pytest.raises(SolveRequestError, match="unknown_type_mismatch"):
        program.solve(replace(request, seeds={"field_tuple": seed}), solver=field.default_program_solver())
    assert (program._next_id, tuple(program._values)) == before


def test_captured_clock_cannot_be_relabelled_after_binding():
    _, field, program, current, request, *_ = mixed_case()
    # value(SSA, at=...) replaces the same record; the frozen equation sees this drift.
    program.value("relabelled-capture", current.n, at=current.next.point)
    before = (program._next_id, tuple(program._values))
    with pytest.raises(SolveRequestError, match="captures"):
        program.solve(request, solver=field.default_program_solver())
    assert (program._next_id, tuple(program._values)) == before


@pytest.mark.parametrize("order,failure", (((0, 1), None), ((1, 0), "iterations"),
                                           ((2, 0, 1), "nonfinite")))
def test_public_native_fixture_resolves_and_emits_without_execution(order, failure):
    import runpy
    from pathlib import Path
    fixture = runpy.run_path(str(Path(__file__).resolve().parents[2] /
        "integration/runtime/test_nonlinear_mixed_field_runtime.py"))
    resolved, _, _ = fixture["prepared_case"](order, failure=failure)
    code = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    solves = [node for node in resolved.time._values if node.op == "solve_spatial_field"]
    assert len(solves) == 2 and all(node.attrs["capture_count"] == 2 for node in solves)
    assert solves[0].attrs["solve_request"]["equation_identity"] == solves[1].attrs["solve_request"]["equation_identity"]
    assert solves[0].attrs["solve_request"]["initialization_identity"] != solves[1].attrs["solve_request"]["initialization_identity"]
    assert code.index("original_field_residual_recheck_failed") < code.index(".report().residual_norm")
    for node in solves:
        assert "auto field_residual_%d_output =" % node.id in code
        assert "copy(*field_residual_%d_output, field_residual_%d_workspace->candidate())" % (node.id,node.id) in code


def test_manufactured_data_close_all_original_equations():
    import runpy
    from pathlib import Path
    import numpy as np
    fixture = runpy.run_path(str(Path(__file__).resolve().parents[2] /
        "integration/runtime/test_nonlinear_mixed_field_runtime.py"))
    for width in (2, 3):
        target, coefficient = fixture["target_means"](width), fixture["parameter_means"]()
        forcing = fixture["original_lhs"](target, coefficient)
        assert forcing.shape == target.shape == (width, 4, 5)
        assert np.isfinite(forcing).all()
        # An independent construction from explicit anisotropic neighbors.
        lap = (np.roll(target, 1, axis=2) - 2 * target + np.roll(target, -1, axis=2)) / .2**2
        lap += (np.roll(target, 1, axis=1) - 2 * target + np.roll(target, -1, axis=1)) / .375**2
        q, v = target[:2]
        expected = np.array([coefficient[0] * q + q**3 + .03*q*v + .02*v**2,
                             1.3*v + .4*v**3 + .02*v*q + .01*q**2]
                            + ([] if width == 2 else [1.4*target[2] + target[2]**3]))
        for row in range(width):
            for column in range(width):
                expected[row] -= fixture["DIFFUSION"][row, column] * lap[column]
        np.testing.assert_allclose(expected, forcing, atol=3e-16, rtol=0)
