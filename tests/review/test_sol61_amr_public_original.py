"""Independent physical authoring and source admission; no author fixture imports."""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import json
import operator

import numpy as np
import pops
import pytest
from pops._ir.elliptic import Reaction, DivCoeffGrad
from pops.amr import (
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.fields import (
    FieldProblem,
    FieldBoundary,
    FieldDiscretization,
    CellCenteredNonlinearCoupled,
    bcs,
)
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR, Uniform
from pops.lib.amr import StateTransfer
from pops.lib.initial import BindArray
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.numerics.terms import SourceTerm
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.solvers import Newton
from pops.time import FailRun, FixedDt, SolveRequestError, every

ROOT = Path(__file__).resolve().parents[2]


def matrix(width):
    return tuple(
        tuple(
            Fraction(10 + i, 10) if i == j else Fraction((-1) ** (i + j) * (2 * i + j + 1), 100)
            for j in range(width)
        )
        for i in range(width)
    )


def authored(width=3, order=None, *, seed=True, boundary="periodic", uniform=False):
    order = tuple(range(width)) if order is None else tuple(order)
    frame = Rectangle("review-anisotropic-box", (0.0, 0.0), (1.0, 1.5)).frame(Cartesian2D())
    physics = pops.Model("physical-consumer", frame=frame)
    response = physics.state("response", components=tuple("r%d" % i for i in range(width)))
    unknowns = tuple(physics.field("v%d" % i) for i in range(width))
    auxiliary = tuple(physics.aux("potential%d" % i) for i in range(width))
    physical_source = physics.source("response_source", on=response, value=auxiliary)
    loads = pops.Model("exact-load", frame=frame)
    forcing = loads.state("forcing", components=tuple("f%d" % i for i in range(width)))
    material = pops.Model("exact-material", frame=frame)
    coefficient = material.state("material", components=("alpha",))
    equations = []
    diffusion = matrix(width)
    for i, unknown in enumerate(unknowns):
        neighbor = unknowns[(i + 1) % width]
        lhs = (
            Reaction(unknown, coefficient[0] + Fraction(i + 2, 10) * ValueExpr(unknown) ** 2)
            + Reaction(unknown, Fraction(3, 100) * ValueExpr(neighbor))
            + Reaction(neighbor, Fraction(2, 100) * ValueExpr(neighbor))
        )
        for j in range(width):
            lhs -= DivCoeffGrad(unknowns[j], float(diffusion[i][j]))
        equations.append(lhs == forcing[i])
    bc = bcs.Periodic() if boundary == "periodic" else bcs.Neumann(0)
    problem = FieldProblem(
        "original-cross-reaction",
        unknowns=tuple(unknowns[i] for i in order),
        equations=tuple(equations[i] for i in order),
        boundaries=tuple(
            FieldBoundary(unknowns[i], bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bc))
            for i in order
        ),
    )
    case = pops.Case("review-original-composite")
    blocks = []
    for name, model, state in (
        ("response", physics, response),
        ("load", loads, forcing),
        ("coefficient", material, coefficient),
    ):
        flux = model.flux(
            "stationary",
            frame=frame,
            state=state,
            components={axis: tuple(0 * q for q in state) for axis in frame.axes},
            waves={axis: tuple(0 * q for q in state) for axis in frame.axes},
        )
        rate = model.rate(
            "balance",
            equation=ddt(state)
            == (-div(flux) + physical_source if model is physics else -div(flux)),
        )
        plan = DiscretizationPlan()
        plan.rates.add(
            rate,
            FiniteVolume(
                flux=flux,
                variables=variables.Conservative(state),
                reconstruction=reconstruction.FirstOrder(),
                riemann=riemann.Rusanov(),
            ),
        )
        block = case.block(name, model)
        case.numerics(plan, block=block)
        case.initials.add(
            InitialCondition(
                state=block[state], value=BindArray(), projection=ConservativeCellAverage()
            )
        )
        blocks.append(block)
    solver = Newton(
        tolerance=2e-10,
        max_iterations=23,
        linear_tolerance=3e-8,
        linear_max_iterations=151,
        restart=41,
    )
    field = case.field(
        problem,
        FieldDiscretization(
            method=CellCenteredNonlinearCoupled(finite_difference_step=2e-7),
            boundaries=(),
            solver=solver,
        ),
    )
    program = pops.Program("one-composite-original")
    state, load, param = (
        program.state(block[handle])
        for block, handle in zip(blocks, (response, forcing, coefficient), strict=True)
    )
    program.store_history("accepted-state", state.n, depth=1)
    request = field.bind_program_inputs(
        program=program,
        values={blocks[1][forcing]: load.n, blocks[2][coefficient]: param.n},
        at=program.stage("original", c=0),
        solver=solver,
    )
    if seed:
        request = replace(
            request,
            seeds={"field_tuple": program.scalar_field("separate-initial-guess", ncomp=width)},
        )
    outcome = program.solve(request, solver=solver)
    observed = field.observe(outcome.consume(action=FailRun()))
    module = physics.module
    carrier = blocks[0][module.field_handle(module.field_spaces()["fields"])]
    published = observed.publish(
        {(carrier, "potential%d" % i): observed[field[unknowns[i]]] for i in range(width)},
        states={blocks[0][response]: state.n},
    )
    rhs = program.rhs(
        state=state.n,
        fields=published,
        terms=[SourceTerm(blocks[0][module.operator_handle("response_source")])],
    )
    program.commit(
        state.next, program.value("source-step", state.n + program.dt * rhs, at=state.next.point)
    )
    for temporal in (load, param):
        program.commit(
            temporal.next,
            program.value("retain-" + temporal.n.name, 1 * temporal.n, at=temporal.next.point),
        )
    program.step_strategy(FixedDt(0.007))
    case.program(program)
    transfer = AMRTransfer()
    for block, handle in zip(blocks, (response, forcing, coefficient), strict=True):
        transfer.state(block[handle], StateTransfer())
    threshold = case.param(RuntimeParam("refinement", default=0.7))
    grid = CartesianGrid(
        frame=frame,
        cells=(10, 6),
        periodic=PeriodicAxes(frame.axes) if boundary == "periodic" else None,
    )
    layout = (
        Uniform(grid)
        if uniform
        else AMR(
            grid=grid,
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(
                rules=(
                    Tag(ValueExpr(blocks[2][coefficient]) > case.value(threshold)),
                    Buffer(cells=1),
                ),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                conflict_policy=ConflictPolicy.REFINE_WINS,
            ),
            regrid=AMRRegrid(schedule=every(1000, clock=program.clock)),
            transfer=transfer,
            execution=AMRExecution.synchronous(),
        )
    )
    return case, layout, program, outcome._token, request


def emit(case, layout):
    resolved = pops.resolve(pops.validate(case), layout=layout)
    return emit_cpp_program(
        resolved.time,
        model=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
        target="amr_system" if isinstance(layout, AMR) else "system",
    ), resolved


def literal(data):
    kind = data["kind"]
    if kind == "rational":
        return int(data["numerator"]) / int(data["denominator"])
    if kind == "binary64":
        return float.fromhex(data["value"])
    return float(data["value"])


def evaluate(node, unknown, captured):
    op = node[0]
    if op == "literal":
        return literal(node[1])
    if op == "unknown":
        return unknown[node[1]]
    if op == "input":
        return captured[node[1]][node[2]]
    if op in ("add", "sub", "mul", "div", "pow"):
        operation = {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "div": operator.truediv,
            "pow": operator.pow,
        }[op]
        return operation(evaluate(node[1], unknown, captured), evaluate(node[2], unknown, captured))
    if op == "neg":
        return -evaluate(node[1], unknown, captured)
    raise AssertionError("unexpected original expression " + repr(node))


@pytest.mark.parametrize("width,order", [(2, (1, 0)), (3, (2, 0, 1)), (5, (4, 1, 3, 0, 2))])
def test_registered_original_ast_preserves_reactions_captures_and_crossflux(width, order):
    _, _, _, token, _ = authored(width, order)
    shape = (4, 5)
    q = np.array([(i + 1) * (0.1 + np.arange(20).reshape(shape) * 0.003) for i in range(width)])
    forcing = np.array(
        [-0.2 + i * 0.02 + np.arange(20).reshape(shape) * 0.001 for i in range(width)]
    )
    alpha = 1.2 + np.arange(20).reshape(shape) * 0.004
    captures = [
        forcing if source.space.components[0] == "f0" else alpha[None]
        for source in token.inputs[2:4]
    ]
    encoded = np.array(
        [evaluate(row, q[list(order)], captures) for row in token.attrs["local_expressions"]]
    )
    expected = np.array(
        [
            alpha * q[i]
            + (i + 2) / 10 * q[i] ** 3
            + 0.03 * q[i] * q[(i + 1) % width]
            + 0.02 * q[(i + 1) % width] ** 2
            - forcing[i]
            for i in order
        ]
    )
    np.testing.assert_allclose(encoded, expected, atol=2e-15, rtol=0)
    diffusion = np.array([literal(row[1]) for row in token.inputs[1].attrs["expressions"]]).reshape(
        width, width
    )
    desired = np.asarray(matrix(width), dtype=float)[np.ix_(order, order)]
    np.testing.assert_array_equal(diffusion, desired)
    lap = (np.roll(q, 1, axis=2) - 2 * q + np.roll(q, -1, axis=2)) / 0.2**2
    lap += (np.roll(q, 1, axis=1) - 2 * q + np.roll(q, -1, axis=1)) / 0.375**2
    residual = encoded - np.einsum("ij,jyx->iyx", diffusion, lap[list(order)])
    physical = (
        expected
        - np.einsum("ij,jyx->iyx", np.asarray(matrix(width), dtype=float), lap)[list(order)]
    )
    np.testing.assert_allclose(residual, physical, atol=3e-15, rtol=0)
    assert (
        np.max(np.abs(residual - (encoded - np.einsum("ji,jyx->iyx", diffusion, lap[list(order)]))))
        > 0.01
    )


@pytest.mark.parametrize(
    "width,order,seed,boundary",
    [
        (2, (1, 0), False, "periodic"),
        (3, (2, 0, 1), True, "neumann"),
        (5, (4, 1, 3, 0, 2), True, "periodic"),
    ],
)
def test_public_partial_amr_one_solve_retained_captures_and_actual_source(
    width, order, seed, boundary, tmp_path
):
    case, layout, _, token, _ = authored(width, order, seed=seed, boundary=boundary)
    code, resolved = emit(case, layout)
    assert code.count("_core->solve(") == 1
    assert "advance_synchronized_hierarchy(dt, _advance_hierarchy, true)" in code
    assert "_level_programs->front().solve(hierarchy_dt)" in code
    assert (
        ".gather(hierarchy_dt)" in code
        and ".observe(hierarchy_dt)" in code
        and ".publish(hierarchy_dt)" in code
    )
    assert code.index("ctx.publish_staged_field_components();") < code.index(
        ".publish(hierarchy_dt)"
    )
    assert "Core::prepare(" in code and "stage_original_field_candidate_collectively" in code
    assert "std::shared_ptr<const amr_original_field_" in code
    assert "retain_hierarchy_field_solver" in code and "original_hierarchy_field_authority" in code
    for index, source in enumerate(token.inputs[2 : 2 + token.attrs["capture_count"]]):
        expected = f"ctx.hierarchy_field_scratch({token.id}, {token.id}, {2 + index}, {len(source.space.components)}, 0, false)"
        assert expected in code
    if seed:
        assert f"ctx.hierarchy_field_scratch({token.id}, {token.id}, 1, {width}, 0, false)" in code
    assert "nonfinite_original_field_residual" in code
    assert "original_amr_field_residual" in code and "response_source" in code
    (tmp_path / "program.cpp").write_text(code)
    (tmp_path / "identity.json").write_text(
        json.dumps(
            {
                "program_ir": resolved.time._ir_hash(),
                "original_equation": token.attrs["solve_request"]["equation_identity"],
                "width": width,
                "order": order,
            }
        )
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"capture_count": 1},
        {"seed_index": 99},
        {"physical_boundary": "invented"},
        {"finite_difference_step": {"kind": "integer", "value": "0"}},
    ],
)
def test_mutated_original_contract_fails_before_codegen(mutation):
    case, layout, program, token, _ = authored()
    program._replace_value(token, attrs={**token.attrs, **mutation})
    with pytest.raises((SolveRequestError, ValueError, TypeError)):
        emit(case, layout)


def test_foreign_capture_is_refused_without_program_mutation():
    _, _, program, _, request = authored()
    foreign = program.scalar_field("wrong-equation-binding", ncomp=3)
    before = (program._next_id, tuple(program._values))
    with pytest.raises(SolveRequestError, match="equation_input_mismatch"):
        program.solve(
            replace(request, equation_inputs={**request.equation_inputs, "capture_0": foreign}),
            solver=Newton(),
        )
    assert (program._next_id, tuple(program._values)) == before


def test_same_equation_separate_seed_preserves_original_identity():
    _, _, program, first, request = authored(seed=False)
    seed = program.scalar_field("distinct-seed", ncomp=3)
    second = program.solve(replace(request, seeds={"field_tuple": seed}), solver=Newton())._token
    assert (
        first.attrs["solve_request"]["equation_identity"]
        == second.attrs["solve_request"]["equation_identity"]
    )
    assert (
        first.attrs["solve_request"]["initialization_identity"]
        != second.attrs["solve_request"]["initialization_identity"]
    )


def test_public_subcycled_layout_is_refused():
    case, layout, *_ = authored()
    with pytest.raises(ValueError, match="explicit synchronous AMR execution"):
        emit(
            case,
            AMR(
                grid=layout.grid,
                hierarchy=layout.hierarchy,
                tagging=layout.tagging,
                regrid=layout.regrid,
                transfer=layout.transfer,
                execution=AMRExecution.subcycled(),
            ),
        )


def test_resealed_original_reaction_is_refused_against_physical_registry():
    from pops.fields._program_nonlinear_problem import (
        _request_data,
        validate_nonlinear_field_request,
    )
    from pops.identity.scalar import scalar_literal
    from pops.time._program.serialization import _json_ready
    from pops.time.solve_request import SolveUnknown

    case, layout, program, token, _ = authored()
    changed = (
        ("add", token.attrs["local_expressions"][0], ("literal", scalar_literal(0.47).to_data())),
        *token.attrs["local_expressions"][1:],
    )
    source = {**token.attrs["source_contract"], "local_expressions": changed}
    token = program._replace_value(
        token, attrs={**token.attrs, "local_expressions": changed, "source_contract": source}
    )
    contract = _request_data(
        program,
        token,
        SolveUnknown("field_tuple", token.inputs[0]),
        _json_ready(token.attrs["solve_request"]["physical_problem"]),
    )
    token = program._replace_value(token, attrs={**token.attrs, "solve_request": contract})
    validate_nonlinear_field_request(program, token)
    with pytest.raises(ValueError, match="registered equations"):
        emit(case, layout)


@pytest.mark.parametrize("order", ((0, 1, 2), (2, 0, 1)))
def test_central_original_derivative_matches_physical_polynomial(order):
    _, _, _, token, _ = authored(3, order)
    q = np.array((0.17, -0.24, 0.31))
    direction = np.array((0.43, -0.16, 0.29))
    alpha = 1.34
    forcing = np.array((0.21, -0.14, 0.53))
    captured = [
        forcing if source.space.components[0] == "f0" else np.array((alpha,))
        for source in token.inputs[2:4]
    ]
    # Independent full local physical Jacobian, with the offdiagonal nonlinear body intact.
    analytical = np.array(
        [
            (alpha + 3 * (i + 2) / 10 * q[i] ** 2 + 0.03 * q[(i + 1) % 3]) * direction[i]
            + (0.03 * q[i] + 0.04 * q[(i + 1) % 3]) * direction[(i + 1) % 3]
            for i in order
        ]
    )
    h = 2e-7 * (1 + np.linalg.norm(q)) / np.linalg.norm(direction)
    plus = np.array(
        [
            evaluate(row, (q + h * direction)[list(order)], captured)
            for row in token.attrs["local_expressions"]
        ]
    )
    minus = np.array(
        [
            evaluate(row, (q - h * direction)[list(order)], captured)
            for row in token.attrs["local_expressions"]
        ]
    )
    np.testing.assert_allclose((plus - minus) / (2 * h), analytical, atol=2e-10, rtol=0)
    assert token.attrs["solve_request"]["derivative"]["scheme"] == "central_full_residual"
    assert token.attrs["capture_count"] == 2
