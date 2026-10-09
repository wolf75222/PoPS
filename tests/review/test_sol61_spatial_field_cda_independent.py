"""Independent public authoring and mathematical reception of exact T3 cda144e9.

No compiled Program or nonlinear native solver is executed by these tests.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pops
import pytest

from pops._ir.elliptic import DivCoeffGrad, Reaction
from pops._ir.handle_expr import ValueExpr
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.fields import CellCenteredNonlinearCoupled, FieldBoundary, FieldDiscretization
from pops.fields import FieldProblem, FieldProblemError, bcs
from pops.fields._program_expression import decode_field_literal
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import Handle, OwnerPath
from pops.projection import ConservativeCellAverage
from pops.solvers import Newton
from pops.time import FailRun, FixedDt, SolveRequestError

from tests.review.test_sol61_spatial_nonlinear_math_oracle import witness

ROOT = Path(__file__).resolve().parents[2]


def public_case(order, *, label="independent", boundary=None):
    data, width = witness(len(order)), len(order)
    frame = CartesianDomain(label + "_domain", lower=(0., 0.), upper=(1., 1.7)).frame(Cartesian2D())
    force_model = pops.Model(label + "_forcing", frame=frame)
    force = force_model.state("rhs", components=tuple(f"load{i}" for i in range(width)))
    parameter_model = pops.Model(label + "_parameters", frame=frame)
    parameter = parameter_model.state("a", components=tuple(f"a{i}" for i in range(width)))
    case = pops.Case(label + "_case")
    block = case.block("right_hand_side", force_model)
    parameter_block = case.block("distinct_coefficients", parameter_model)
    unknowns = tuple(Handle(f"unknown{i}", kind="field", owner=OwnerPath.model(label + "_tuple"))
                     for i in range(width))
    equations = []
    for i, unknown in enumerate(unknowns):
        coefficient = float(data.linear[i]) + float(data.cubic[i]) * ValueExpr(unknown)**2
        for j in range(width):
            if data.product[i, j]:
                coefficient = coefficient + parameter[i] * float(data.product[i, j]) * ValueExpr(unknowns[j])
        lhs = Reaction(unknown, coefficient)
        for j in range(width):
            if data.square[i, j]:
                lhs = lhs + Reaction(unknowns[j], float(data.square[i, j]) * ValueExpr(unknowns[j]))
            if data.diffusion[i, j]:
                lhs = lhs - DivCoeffGrad(unknowns[j], float(data.diffusion[i, j]))
        equations.append(lhs == force[i])
    problem = FieldProblem(label + "_problem", unknowns=tuple(unknowns[i] for i in order),
        equations=tuple(equations[i] for i in order), boundaries=tuple(FieldBoundary(
            unknowns[i], bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), boundary or bcs.Periodic()))
            for i in order))
    solver = Newton(tolerance=1e-11, linear_tolerance=1e-9, linear_max_iterations=150, restart=60)
    field = case.field(problem, FieldDiscretization(method=CellCenteredNonlinearCoupled(
        finite_difference_step=1e-7), boundaries=(), solver=solver))
    program = pops.Program(label + "_program")
    load, coefficients = program.state(block[force]), program.state(parameter_block[parameter])
    request = field.bind_program_inputs(program=program,
        values={block[force]: load.n, parameter_block[parameter]: coefficients.n},
        at=load.next.point, solver=solver)
    return case, field, program, request, solver, load, coefficients, data, frame, unknowns


def finish(args, *, output=None, mutate=None):
    case, field, program, request, solver, load, coefficients, _, frame, unknowns = args
    first = program.solve(request, solver=solver).consume(action=FailRun())
    seed = program.value("second_seed", .8 * first[0], at=load.next.point)
    second = program.solve(replace(request, seeds={"field_tuple": seed}), solver=solver).consume(action=FailRun())
    for run, result in enumerate((first, second)):
        observed = field.observe(result)
        for i, unknown in enumerate(unknowns):
            program.store_history(f"solution{run}_{i}", observed[field[unknown]], depth=1)
    for current in (load, coefficients):
        program.commit(current.next, program.value("accepted_capture", 1 * current.n, at=current.next.point))
    program.step_strategy(FixedDt(.01))
    if mutate is not None:
        mutate(program)
    case.program(program)
    for state in (load.n.state_ref, coefficients.n.state_ref):
        case.initials.add(InitialCondition(state=state, value=BindArray(), projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(5, 4), periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    if output is not None:
        Path(output).write_text(code)
    return code, resolved


def ast_value(node, candidate, captures):
    op = node[0]
    if op == "literal":
        return float(decode_field_literal(node[1]).to_python())
    if op == "unknown":
        return candidate[node[1]]
    if op == "input":
        return captures[node[1]][node[2]]
    if op == "neg":
        return -ast_value(node[1], candidate, captures)
    a, b = (ast_value(child, candidate, captures) for child in node[1:])
    return {"add": np.add, "sub": np.subtract, "mul": np.multiply,
            "div": np.divide, "pow": np.power}[op](a, b)


@pytest.mark.parametrize("order", ((0, 1), (2, 0, 1), (4, 1, 3, 0, 2)))
def test_original_closed_ast_matches_independent_reaction_and_cross_diffusion(order):
    args = public_case(order)
    _, _, program, request, solver, load, coefficients, data, _, _ = args
    token = program.solve(request, solver=solver)._token
    captures = [data.rhs if part.state_ref == load.n.state_ref else data.capture
                for part in token.inputs[2:2 + token.attrs["capture_count"]]]
    for candidate in (data.exact, -.2 - .3 * data.exact, .7 * data.exact + .1):
        selected = candidate[list(order)]
        local = np.array([ast_value(node, selected, captures) for node in token.attrs["local_expressions"]])
        diffusion = np.array([ast_value(node, selected, captures)
                              for node in token.inputs[1].attrs["expressions"]]).reshape(len(order), len(order))
        actual = local - diffusion @ selected @ data.laplacian.T
        np.testing.assert_allclose(actual, data.residual(candidate)[list(order)], atol=2e-14, rtol=2e-14)
    assert token.attrs["contract"] == "pops.spatial-field-residual@1"
    assert token.attrs["solve_request"]["derivative"]["scheme"] == "central_full_residual"


@pytest.mark.parametrize("label", ("plain", "arbitrary_rename"))
def test_five_unknowns_two_captures_and_two_distinct_actual_solve_buffers(label):
    code, resolved = finish(public_case((4, 1, 3, 0, 2), label=label))
    solves = [node for node in resolved.time._values if node.op == "solve_spatial_field"]
    assert len(solves) == 2
    assert all(node.attrs["ncomp"] == 5 and node.attrs["capture_count"] == 2 for node in solves)
    first, second = (node.attrs["solve_request"] for node in solves)
    assert first["equation_identity"] == second["equation_identity"]
    assert first["initialization_identity"] != second["initialization_identity"]
    assert "apply_general_field<pops::kNativeDimension, 5, 25>" in code
    assert "PreparedSpatialResidual" in code and "original_field_residual_recheck_failed" in code
    assert "block_inverse" not in code and "condensed" not in code
    for node in solves:
        assert f"auto field_residual_{node.id}_output =" in code
        assert f"copy(*field_residual_{node.id}_output, field_residual_{node.id}_workspace->candidate())" in code
        assert f"field_residual_{node.id}_capture_0" in code
        assert f"field_residual_{node.id}_capture_1" in code


@pytest.mark.parametrize("attack", ("capture_swap", "foreign_seed", "wrong_outputs", "wrong_derivative"))
def test_public_request_refusals_preserve_authoring_transaction(attack):
    from pops.time import DerivativeStrategy
    _, _, program, request, solver, *_ = public_case((0, 1, 2))
    before = (program._next_id, program._next_region, program._serialize())
    with pytest.raises((SolveRequestError, ValueError)):
        if attack == "capture_swap":
            inputs = dict(request.equation_inputs)
            inputs["capture_0"], inputs["capture_1"] = inputs["capture_1"], inputs["capture_0"]
            request = replace(request, equation_inputs=inputs)
        elif attack == "foreign_seed":
            foreign = pops.Program("foreign")
            request = replace(request, seeds={"field_tuple": foreign.scalar_field("seed", ncomp=3)})
        elif attack == "wrong_outputs":
            request = replace(request, outputs=("different_output",))
        else:
            request = replace(request, derivative=DerivativeStrategy("exact"))
        program.solve(request, solver=solver)
    assert (program._next_id, program._next_region, program._serialize()) == before


def test_physical_boundary_refusal_before_accepted_program_exists():
    with pytest.raises(FieldProblemError, match="periodic or homogeneous Neumann"):
        public_case((0, 1), boundary=bcs.Neumann(1))


@pytest.mark.parametrize("attack", ("literal_unit", "fd_policy"))
def test_resealed_semantics_must_match_registered_physics_and_method(attack):
    from pops.fields._program_nonlinear_problem import _request_data, validate_nonlinear_field_request
    from pops.identity.scalar import scalar_data, scalar_literal
    from pops.time._program.serialization import _json_ready
    from pops.time.solve_request import SolveUnknown

    def corrupt(program):
        token = next(node for node in program._values if node.op == "solve_spatial_field")
        source, attrs = dict(token.attrs["source_contract"]), dict(token.attrs)
        if attack == "literal_unit":
            literal = {**scalar_literal(0).to_data(), "unit": "metre"}
            changed = (("add", attrs["local_expressions"][0], ("literal", literal)),
                       *attrs["local_expressions"][1:])
            attrs["local_expressions"] = source["local_expressions"] = changed
        else:
            attrs["finite_difference_step"] = source["finite_difference_step"] = scalar_data(2e-7)
        token = program._replace_value(token, attrs={**attrs, "source_contract": source})
        old = token.attrs["solve_request"]
        resealed = _request_data(program, token, SolveUnknown("field_tuple", token.inputs[0]),
                                _json_ready(old["physical_problem"]))
        token = program._replace_value(token, attrs={**token.attrs, "solve_request": resealed})
        validate_nonlinear_field_request(program, token)  # self-consistent forged seal

    error_type = TypeError if attack == "literal_unit" else ValueError
    reason = "explicit unit-system conversion" if attack == "literal_unit" else "registered equations/method/boundaries"
    with pytest.raises(error_type, match=reason):
        finish(public_case((0, 1)), mutate=corrupt)


def test_uniform_only_realization_refuses_amr_target():
    _, resolved = finish(public_case((0, 1)))
    with pytest.raises((ValueError, NotImplementedError), match="AMR|Uniform|hierarchy"):
        emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
                         target="amr_system")


def test_selected_native_scaled_central_jvp_matches_original_analytical_jacobian():
    data = witness()
    values = data.exact + .2
    direction = np.sin(np.arange(values.size) + .3).reshape(values.shape)
    step = 1e-7 * max(1., np.linalg.norm(values)) / np.linalg.norm(direction)
    central = (data.residual(values + step * direction) - data.residual(values - step * direction)) / (2 * step)
    exact = (data.jacobian(values) @ direction.ravel()).reshape(values.shape)
    np.testing.assert_allclose(central, exact, atol=3e-8, rtol=3e-8)
