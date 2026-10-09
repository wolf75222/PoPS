"""Independent State-D identity/registration attacks and original stencil math.

Only public physical construction is shared; no author's AST evaluator or oracle.
No installed/native solve, JIT, or MPI is executed.
"""

from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from pops.solvers import Newton
from pops.math import ValueExpr
from pops.fields import FieldProblemError, CellCenteredNonlinearCoupled
from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request, _request_data
from pops.time import SolveRequestError
from pops.time.solve_request import SolveUnknown
from pops.time.values import ProgramValue
from pops.time._program.serialization import _json_ready
from pops._frozen_data import freeze_containers
from pops.fields._identity import field_identity
import test_sol61_amr_public_original as physical

ROOT = Path(__file__).resolve().parents[2]
FACE = "pops.field.face-mean.arithmetic@1"


def captured_case(
    *, width=3, order=(2, 0, 1), uniform=True, policy="Arithmetic@1", unknown=False, jacobi=False
):
    declarations = []
    state_constructor = physical.pops.Model.state

    def declare(model, *args, **kwargs):
        state = state_constructor(model, *args, **kwargs)
        declarations.append(state)
        return state

    original_diffusion = physical.DivCoeffGrad

    def diffusion(field, coefficient):
        load, alpha = declarations[-2:]
        value = coefficient * (Fraction(3, 2) + alpha[0] / 3) + (load[0] - 2 * alpha[0]) / 7
        if unknown:
            value += ValueExpr(field) ** 2
        return original_diffusion(field, value)

    with ExitStack() as stack:
        stack.enter_context(patch.object(physical.pops.Model, "state", declare))
        stack.enter_context(patch.object(physical, "DivCoeffGrad", diffusion))
        stack.enter_context(
            patch.object(
                physical,
                "CellCenteredNonlinearCoupled",
                partial(CellCenteredNonlinearCoupled, face_policy=policy),
            )
        )
        if jacobi:
            stack.enter_context(
                patch.object(
                    physical, "Newton", partial(Newton, right_preconditioner="SpatialBasisJacobi@1")
                )
            )
        return physical.authored(width, order, uniform=uniform)


def clone(value, *, attrs=None, inputs=None, point=None):
    return ProgramValue(
        value.prog,
        value.id,
        value.vtype,
        value.op,
        value.inputs if inputs is None else inputs,
        value.attrs if attrs is None else attrs,
        value.name,
        value.block,
        space=value.space,
        source_location=value.source_location,
        field_context=value.field_context,
        region=value.region,
        state_ref=value.state_ref,
        point=value.point if point is None else point,
        provenance=value.provenance,
    )


def evaluate(node, captures):
    op = node[0]
    if op == "literal":
        data = node[1]
        if data["kind"] == "rational":
            return Fraction(int(data["numerator"]), int(data["denominator"]))
        if data["kind"] == "integer":
            return Fraction(int(data["value"]))
        return Fraction(float.fromhex(data["value"]))
    if op == "input":
        return captures[node[1]][node[2]]
    if op == "neg":
        return -evaluate(node[1], captures)
    a, b = evaluate(node[1], captures), evaluate(node[2], captures)
    return {
        "add": lambda: a + b,
        "sub": lambda: a - b,
        "mul": lambda: a * b,
        "div": lambda: a / b,
        "pow": lambda: a ** int(b),
    }[op]()


@pytest.fixture(scope="module")
def witness():
    return captured_case()


def test_actual_two_state_coefficient_inputs_and_exact_ast(witness):
    _, _, program, token, _ = witness
    validate_nonlinear_field_request(program, token)
    coefficient = token.inputs[1]
    captures = token.inputs[2 : 2 + token.attrs["capture_count"]]
    assert len(coefficient.attrs["field_dependencies"]) == 2
    assert len(coefficient.inputs) == len(captures)
    assert all(a is b for a, b in zip(coefficient.inputs, captures, strict=True))
    assert coefficient.point == token.point == token.inputs[0].point
    values = []
    for capture in captures:
        values.append(
            [Fraction(-3, 2)]
            if tuple(capture.space.components) == ("alpha",)
            else [Fraction(5, 7)] + [Fraction(-11, 13)] * (len(capture.space.components) - 1)
        )
    alpha, load = Fraction(-3, 2), Fraction(5, 7)
    order = (2, 0, 1)
    for i in range(3):
        for j in range(3):
            declared = float(physical.matrix(3)[order[i]][order[j]])
            expected = Fraction(declared) * (Fraction(3, 2) + alpha / 3) + (load - 2 * alpha) / 7
            assert evaluate(coefficient.attrs["expressions"][3 * i + j], values) == expected
    assert token.attrs["contract"] == "pops.spatial-field-residual@2"
    assert token.attrs["solve_request"]["schema_version"] == 2
    assert token.attrs["solve_request"]["realization"] == {"coefficient_face_policy": FACE}
    assert program._serialize()["version"] == 10


@pytest.mark.parametrize(
    "mutation",
    [
        "reverse",
        "clone_equal",
        "missing_dep",
        "fake_dep",
        "point",
        "input_id",
        "null_policy",
        "private_policy",
    ],
)
def test_actual_coefficient_identity_refuses_before_mutation(witness, mutation):
    _, _, program, token, _ = witness
    coefficient = token.inputs[1]
    attrs = deepcopy(_json_ready(coefficient.attrs))
    inputs = coefficient.inputs
    point = coefficient.point
    if mutation == "reverse":
        inputs = tuple(reversed(inputs))
    elif mutation == "clone_equal":
        inputs = (clone(inputs[0]), *inputs[1:])
    elif mutation == "missing_dep":
        attrs["field_dependencies"] = ()
    elif mutation == "fake_dep":
        attrs["field_dependencies"] = tuple(reversed(attrs["field_dependencies"]))
    elif mutation == "point":
        point = program.stage("stale-D", c=1)
    elif mutation == "input_id":
        victim = clone(inputs[0])
        object.__setattr__(victim, "id", inputs[0].id + 1000)
        inputs = (victim, *inputs[1:])
    elif mutation == "null_policy":
        attrs["coefficient_face_policy"] = None
    else:
        attrs["coefficient_face_policy"] = "pops.field.face-mean.arithmetic@2"
    before = program._ir_hash()
    forged = clone(coefficient, attrs=attrs, inputs=inputs, point=point)
    bad = clone(token, inputs=(token.inputs[0], forged, *token.inputs[2:]))
    with pytest.raises((SolveRequestError, ValueError)):
        validate_nonlinear_field_request(program, bad)
    assert program._ir_hash() == before


@pytest.mark.parametrize(
    "policy", [None, "pops.field.face-mean.harmonic@1", "pops.field.face-mean.arithmetic@2"]
)
def test_resigned_policy_refuses_same_original_registration(witness, policy):
    _, _, program, token, _ = witness
    attrs = deepcopy(_json_ready(token.attrs))
    source = deepcopy(attrs["source_contract"])
    source["coefficient_face_policy"] = policy
    attrs["source_contract"] = freeze_containers(source)
    attrs["coefficient_face_policy"] = policy
    coeff_attrs = {**token.inputs[1].attrs, "coefficient_face_policy": policy}
    coefficient = clone(token.inputs[1], attrs=coeff_attrs)
    forged = clone(token, attrs=attrs, inputs=(token.inputs[0], coefficient, *token.inputs[2:]))
    if policy is None:
        attrs["contract"] = "pops.spatial-field-residual@1"
        forged = clone(forged, attrs=attrs)
    actual = attrs["solve_request"]
    unknown = SolveUnknown(actual["unknowns"][0]["name"], forged.inputs[0])
    # Consistent self-digests do not supply a missing or unknown physical realization.
    request = _request_data(program, forged, unknown, _json_ready(actual["physical_problem"]))
    forged = clone(forged, attrs={**forged.attrs, "solve_request": request})
    with pytest.raises((SolveRequestError, ValueError)):
        validate_nonlinear_field_request(program, forged)


def test_captured_d_without_policy_and_candidate_d_are_explicit_refusals():
    with pytest.raises(FieldProblemError, match="requires explicit"):
        captured_case(policy=None)
    with pytest.raises(FieldProblemError, match="unknown-dependent D.*full spatial nonlinear JVP"):
        captured_case(unknown=True)


@pytest.mark.parametrize("bad", [None, True, "Arithmetic@2", "Harmonic@1"])
def test_registered_method_change_refuses_after_resolution(witness, bad):
    case, layout, _, _, _ = witness
    resolved = physical.pops.resolve(physical.pops.validate(case), layout=layout)
    registration = next(iter(resolved.program_field_plans.values()))
    method = registration.discretization.method
    previous, identity = method.face_policy, registration.identity
    with pytest.raises(RuntimeError, match="is frozen"):
        method.face_policy = bad
    try:
        object.__setattr__(method, "face_policy", bad)
        if bad is None:
            # Re-digest valid default metadata, then require the original registered
            # program/realization to agree; an identity-only check is insufficient.
            object.__setattr__(
                registration,
                "identity",
                field_identity("resolved-program-field", registration.to_data(False)),
            )
        with pytest.raises((ValueError, TypeError)):
            registration.validate_program(resolved.time)
    finally:
        object.__setattr__(method, "face_policy", previous)
        object.__setattr__(registration, "identity", identity)
    registration.validate_program(resolved.time)


def test_uniform_preparation_before_newton_and_no_live_capture_reads(witness):
    case, layout, _, token, _ = witness
    cpp, _ = physical.emit(case, layout)
    prepare = "prepare_general_field_coefficients<pops::kNativeDimension, 3, 9, false>"
    callback = f"auto field_residual_{token.id}_evaluate"
    assert cpp.index(prepare) < cpp.index(callback)
    assert "apply_general_field<pops::kNativeDimension, 3, 9, true>" in cpp
    body = cpp.split(callback, 1)[1].split(
        f"pops::SolveReport field_residual_{token.id}_report", 1
    )[0]
    assert "_capture_" in body
    assert "field_source_" not in body
    assert "nonfinite_original_field_residual" in body


def test_scalar_amr_arithmetic_uses_original_guarded_operator():
    case, layout, program, token, _ = captured_case(width=1, order=(0,), uniform=False)
    cpp, _ = physical.emit(case, layout)
    assert program._serialize()["version"] == 10
    assert "ctx.hierarchy_field_assembly" in cpp and "input" in cpp
    assert cpp.index("ctx.hierarchy_field_assembly") < cpp.index("_Core::prepare(")
    core = (ROOT / "include/pops/runtime/program/prepared_amr_field_residual.hpp").read_text()
    # Frozen@1 prepares its captured operator before recording that preparation.
    # PerCandidate@1 has a separate allocation-only branch and generation witness;
    # its earlier textual occurrence must not stand in for the legacy control flow.
    frozen = core.split(
        "if (coefficient_evaluation == AmrFieldCoefficientEvaluation::kFrozen)", 1
    )[1]
    assert frozen.index("op.prepare_original_field_operator();") < frozen.index(
        "result->coefficient_generation_"
    )
    provider = (
        ROOT / "include/pops/numerics/elliptic/nd/prepared_composite_general_field.hpp"
    ).read_text()
    assert "scalar.apply_linear_composite(guarded || options_.coefficients != n," in provider
    assert "prepare_coefficients_impl_(true, false)" in provider
    assert token.inputs[1].attrs["coefficient_admissibility"] == "finite_general"


# Autonomous exact arithmetic; no PoPS expression evaluator or discretization helper.
def original_image(shape, lengths, coefficients, q):
    import itertools

    cells = tuple(itertools.product(*(range(n) for n in shape)))
    width = len(next(iter(q.values())))
    image = {cell: [Fraction(0)] * width for cell in cells}
    for cell in cells:
        for axis, n in enumerate(shape):
            lower = list(cell)
            lower[axis] = (lower[axis] - 1) % n
            lower = tuple(lower)
            upper = list(cell)
            upper[axis] = (upper[axis] + 1) % n
            upper = tuple(upper)
            spacing = lengths[axis] / n
            for i in range(width):
                for j in range(width):
                    low = (coefficients[lower][i][j] + coefficients[cell][i][j]) / 2
                    high = (coefficients[cell][i][j] + coefficients[upper][i][j]) / 2
                    image[cell][i] -= (
                        high * (q[upper][j] - q[cell][j]) - low * (q[cell][j] - q[lower][j])
                    ) / spacing**2
    return {cell: tuple(row) for cell, row in image.items()}


@pytest.mark.parametrize(
    "shape,lengths,width",
    [
        ((7,), (Fraction(5, 3),), 1),
        ((4, 3), (Fraction(5, 3), Fraction(7, 4)), 2),
        ((3, 2, 4), (Fraction(5, 3), Fraction(7, 4), Fraction(11, 5)), 5),
    ],
)
def test_original_frozen_d_full_jvp_conservation_and_component_permutation(shape, lengths, width):
    import itertools

    cells = tuple(itertools.product(*(range(n) for n in shape)))
    matrix = tuple(
        tuple(
            Fraction((i + 1) * ((-1) ** j) * (j + 2), 11)
            if j != width - 1 or width == 1
            else Fraction(0)
            for j in range(width)
        )
        for i in range(width)
    )
    # Signed, nonsymmetric and (width>1) singular finite matrices are mathematical
    # inputs, never an SPD or native convergence certificate.
    coefficients = {
        cell: tuple(
            tuple(
                matrix[i][j] * Fraction((-1) ** sum(cell) * (sum(cell) + 1), 6)
                for j in range(width)
            )
            for i in range(width)
        )
        for cell in cells
    }
    q = {
        cell: tuple(
            Fraction(
                (i + 1) * (1 + sum((a + 2) * c for a, c in enumerate(cell))) + (-1) ** sum(cell), 17
            )
            for i in range(width)
        )
        for cell in cells
    }
    d = {
        cell: tuple(Fraction((-1) ** (i + sum(cell)) * (i + 2), 13) for i in range(width))
        for cell in cells
    }
    image = original_image(shape, lengths, coefficients, q)
    assert any(value for row in image.values() for value in row)
    assert all(sum(image[cell][i] for cell in cells) == 0 for i in range(width))
    order = tuple(reversed(range(width)))
    perm_d = {
        cell: tuple(tuple(coefficients[cell][i][j] for j in order) for i in order) for cell in cells
    }
    perm_q = {cell: tuple(q[cell][i] for i in order) for cell in cells}
    perm_image = original_image(shape, lengths, perm_d, perm_q)
    assert all(perm_image[cell] == tuple(image[cell][i] for i in order) for cell in cells)

    def residual(candidate):
        spatial = original_image(shape, lengths, coefficients, candidate)
        return {
            cell: tuple(
                spatial[cell][i]
                + candidate[cell][i] ** 3
                + sum(Fraction(i + j + 1, 19) * candidate[cell][j] ** 2 for j in range(width))
                for i in range(width)
            )
            for cell in cells
        }

    h = Fraction(1, 100003)
    plus = {cell: tuple(q[cell][i] + h * d[cell][i] for i in range(width)) for cell in cells}
    minus = {cell: tuple(q[cell][i] - h * d[cell][i] for i in range(width)) for cell in cells}
    fplus, fminus = residual(plus), residual(minus)
    spatial_d = original_image(shape, lengths, coefficients, d)
    for cell in cells:
        for i in range(width):
            finite_difference = (fplus[cell][i] - fminus[cell][i]) / (2 * h)
            original = (
                spatial_d[cell][i]
                + 3 * q[cell][i] ** 2 * d[cell][i]
                + sum(Fraction(2 * (i + j + 1), 19) * q[cell][j] * d[cell][j] for j in range(width))
            )
            assert finite_difference == original + h**2 * d[cell][i] ** 3
    # Transposing the nonsymmetric law is not a valid coupled contraction.
    if width > 1:
        transposed = {
            cell: tuple(tuple(coefficients[cell][j][i] for j in range(width)) for i in range(width))
            for cell in cells
        }
        assert original_image(shape, lengths, transposed, q) != image


def test_scalar_arithmetic_law_is_distinct_from_harmonic_without_tolerance_change():
    shape = (3,)
    lengths = (Fraction(1),)
    coefficients = {(0,): ((Fraction(1),),), (1,): ((Fraction(3),),), (2,): ((Fraction(7),),)}
    q = {(0,): (Fraction(0),), (1,): (Fraction(2),), (2,): (Fraction(-1),)}
    arithmetic = original_image(shape, lengths, coefficients, q)
    harmonic = {}
    for i in range(3):
        lo, hi = (i - 1) % 3, (i + 1) % 3
        center = coefficients[(i,)][0][0]
        dl, dh = coefficients[(lo,)][0][0], coefficients[(hi,)][0][0]
        low, high = 2 * center * dl / (center + dl), 2 * center * dh / (center + dh)
        harmonic[(i,)] = (
            -(high * (q[(hi,)][0] - q[(i,)][0]) - low * (q[(i,)][0] - q[(lo,)][0])) * 9,
        )
    assert harmonic != arithmetic
    assert (
        sum(row[0] for row in harmonic.values()) == sum(row[0] for row in arithmetic.values()) == 0
    )


def test_captured_d_and_jacobi_policies_are_both_exact_and_materialized_first():
    case, layout, program, token, _ = captured_case(
        width=2, order=(1, 0), uniform=False, jacobi=True
    )
    validate_nonlinear_field_request(program, token)
    assert token.attrs["solve_request"]["realization"] == {
        "coefficient_face_policy": FACE,
        "right_preconditioner": "pops.amr.original-spatial-jacobi.basis-response@1",
    }
    cpp, resolved = physical.emit(case, layout)
    assert program._serialize()["version"] == resolved.time._serialize()["version"] == 10
    assert "AmrFieldRightPreconditioner::kSpatialBasisJacobi" in cpp
    assert ".gather(hierarchy_dt)" in cpp and ".solve(hierarchy_dt)" in cpp
    assert cpp.index(".gather(hierarchy_dt)") < cpp.index(".solve(hierarchy_dt)")
    assert "input0(index, 0)" in cpp and "input1(index, 0)" in cpp
    assert cpp.count("_core->solve(") == 1
