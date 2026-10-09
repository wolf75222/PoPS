"""Original D(State captures) authoring and independent conservative stencil math."""
from fractions import Fraction
import json
from pathlib import Path

import pytest
from pops.fields import CellCenteredNonlinearCoupled, FieldProblemError
from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
from pops.time import SolveRequestError
from pops.math import ValueExpr
import test_sol61_amr_public_original as physical

ROOT = Path(__file__).resolve().parents[2]
FACE = "pops.field.face-mean.arithmetic@1"


def authored(monkeypatch, *, uniform=False, width=3, order=None, policy="Arithmetic@1", unknown=False):
    # Existing independently authored model/circuit-free FieldProblem. Change
    # only its explicitly declared diffusion body and face realization, before
    # the Program is authored. Captures still come from the exact Case State.
    declarations = []
    state_constructor = physical.pops.Model.state
    def declare(model, *args, **kwargs):
        state = state_constructor(model, *args, **kwargs)
        declarations.append(state)
        return state
    monkeypatch.setattr(physical.pops.Model, "state", declare)
    principal = physical.DivCoeffGrad
    def diffusion(field, coefficient):
        parameter = declarations[-1]
        value = ValueExpr(field) if unknown else parameter[0]
        return principal(field, coefficient * (1 + value))
    monkeypatch.setattr(physical, "DivCoeffGrad", diffusion)
    method = physical.CellCenteredNonlinearCoupled
    monkeypatch.setattr(physical, "CellCenteredNonlinearCoupled", lambda **kwargs:
                        method(**kwargs, face_policy=policy))
    return physical.authored(width, order, uniform=uniform)


@pytest.mark.parametrize("uniform,width,order", [(True,1,(0,)),(True,2,(1,0)),(False,3,(2,0,1)),(False,5,(4,1,3,0,2))])
def test_original_captured_d_reaches_actual_coefficient_kernel_and_solver(monkeypatch, uniform, width, order):
    case, layout, program, token, _ = authored(monkeypatch, uniform=uniform, width=width, order=order)
    validate_nonlinear_field_request(program, token)
    coefficient = token.inputs[1]
    captures = token.inputs[2:2+token.attrs["capture_count"]]
    assert coefficient.inputs == captures
    assert len(coefficient.attrs["field_dependencies"]) == 1
    assert token.attrs["contract"] == "pops.spatial-field-residual@2"
    assert token.attrs["coefficient_face_policy"] == FACE
    assert token.attrs["solve_request"]["schema_version"] == 2
    assert token.attrs["solve_request"]["realization"] == {"coefficient_face_policy": FACE}
    assert program._serialize()["version"] == 10
    code, resolved = physical.emit(case, layout)
    assert resolved.time._serialize()["version"] == 10
    parameter_index = next(i for i, source in enumerate(captures)
                           if tuple(source.space.components) == ("alpha",))
    assert "field_source_" in code and f"input{parameter_index}(index, 0)" in code
    assert "field expression inputs require exact layout/distribution identity" in code
    assert "std::isfinite(value_" in code
    if uniform:
        assert f"apply_general_field<pops::kNativeDimension, {width}, {width*width}, true>" in code
        assert "prepare_general_field_coefficients" in code
    else:
        assert "ctx.hierarchy_field_assembly" in code
        assert "ctx.prepare_spatial_collectively" in code
        assert "retain_hierarchy_field_solver" in code
        assert code.count("_core->solve(") == 1
    assert "nonfinite_original_field_residual" in code
    # Actual AST evaluation remains D[permuted i,j]*(1+captured alpha).
    diffusion = physical.matrix(width)
    alpha = -Fraction(3,2)
    inputs = [[float(alpha)] if index == parameter_index else [0]*len(source.space.components)
              for index, source in enumerate(captures)]
    for i in range(width):
        for j in range(width):
            observed = physical.evaluate(coefficient.attrs["expressions"][i*width+j], [], inputs)
            assert observed == pytest.approx(float(diffusion[order[i]][order[j]]*(1+alpha)))


def test_captured_d_requires_explicit_face_realization(monkeypatch):
    with pytest.raises(FieldProblemError, match="requires explicit"):
        authored(monkeypatch, policy=None)


def test_captured_d_and_spatial_jacobi_are_separate_authentic_choices(monkeypatch):
    from pops.solvers import Newton
    monkeypatch.setattr(physical, "Newton", lambda **controls:
                        Newton(**controls, right_preconditioner="SpatialBasisJacobi@1"))
    case, layout, program, token, _ = authored(monkeypatch, width=2, order=(1,0))
    validate_nonlinear_field_request(program, token)
    assert token.attrs["solve_request"]["realization"] == {
        "coefficient_face_policy": FACE,
        "right_preconditioner": "pops.amr.original-spatial-jacobi.basis-response@1",
    }
    code, resolved = physical.emit(case, layout)
    assert resolved.time._serialize()["version"] == 10
    assert "SpatialBasisJacobi" in code
    assert "ctx.hierarchy_field_assembly" in code
    assert code.count("_core->solve(") == 1


def test_unknown_dependent_diffusion_is_not_a_frozen_capture(monkeypatch):
    with pytest.raises(FieldProblemError, match="unknown-dependent D"):
        authored(monkeypatch, unknown=True)


@pytest.mark.parametrize("value", [True, 1, "Arithmetic", "Harmonic@1", "Arithmetic@2", {}])
def test_invalid_face_realization_refused(value):
    with pytest.raises(ValueError, match="face_policy"):
        CellCenteredNonlinearCoupled(finite_difference_step=1e-6, face_policy=value)


def test_legacy_literal_and_method_metadata_remain_unextended():
    case, layout, program, token, _ = physical.authored()
    coefficient = token.inputs[1]
    assert coefficient.inputs == ()
    assert coefficient.attrs["field_dependencies"] == ()
    assert all(expression[0] == "literal" for expression in coefficient.attrs["expressions"])
    assert token.attrs["contract"] == "pops.spatial-field-residual@1"
    assert token.attrs["solve_request"]["schema_version"] == 1
    assert "coefficient_face_policy" not in token.attrs
    assert program._serialize()["version"] == 8
    method = CellCenteredNonlinearCoupled(finite_difference_step=1e-6)
    assert "coefficient_face_policy" not in method.options()
    assert "contract" in method.options()


@pytest.mark.parametrize("mutation", ["swapped_captures", "missing_dependency", "unknown_mean", "wrong_point"])
def test_foreign_coefficient_binding_refused_before_emission(monkeypatch, mutation):
    _, _, program, token, _ = authored(monkeypatch, width=2, order=(1,0))
    coefficient = token.inputs[1]
    attrs = dict(coefficient.attrs)
    inputs = coefficient.inputs
    if mutation == "swapped_captures":
        inputs = tuple(reversed(inputs))
    elif mutation == "missing_dependency":
        attrs["field_dependencies"] = ()
    elif mutation == "unknown_mean":
        attrs["coefficient_face_policy"] = "pops.field.face-mean.harmonic@1"
    else:
        coefficient = program._replace_value(coefficient, point=program.stage("wrong", c=1))
    if mutation != "wrong_point":
        coefficient = program._replace_value(coefficient, attrs=attrs)
        object.__setattr__(coefficient, "inputs", inputs)
    token = program._replace_value(token)
    object.__setattr__(token, "inputs", (token.inputs[0], coefficient, *token.inputs[2:]))
    with pytest.raises((SolveRequestError, ValueError, TypeError)):
        validate_nonlinear_field_request(program, token)


@pytest.mark.parametrize("matrix", [
    ((Fraction(2),Fraction(1,3)),(Fraction(-1,5),Fraction(1))),
    ((Fraction(-2),Fraction(0)),(Fraction(1,7),Fraction(-1))),
    ((Fraction(0),Fraction(-1,3)),(Fraction(2,5),Fraction(0))),
])
def test_signed_discontinuous_tensor_arithmetic_flux_conserves_original_law(matrix):
    # Independent exact finite-volume face oracle with nonuniform captured D,
    # signed/nonsymmetric/zero entries; q is a genuine two-component field.
    factors = (Fraction(1),Fraction(-1,2),Fraction(3,2),Fraction(0))
    q = ((Fraction(1,3),Fraction(2)),(Fraction(-1),Fraction(1,5)),
         (Fraction(2,7),Fraction(-3,2)),(Fraction(1),Fraction(0)))
    h = Fraction(1,4)
    fluxes = []
    for cell in range(4):
        right = (cell+1)%4
        face = [[(factors[cell]+factors[right])*matrix[i][j]/2 for j in range(2)] for i in range(2)]
        fluxes.append(tuple(-sum(face[i][j]*(q[right][j]-q[cell][j])/h for j in range(2)) for i in range(2)))
    original = [tuple((fluxes[cell][i]-fluxes[(cell-1)%4][i])/h for i in range(2)) for cell in range(4)]
    assert all(sum(h*original[cell][i] for cell in range(4)) == 0 for i in range(2))
    assert any(value != 0 for row in original for value in row)
    # This is a discrete law check, never a native/FAC convergence claim.
    assert json.dumps([[str(x) for x in row] for row in original])


def test_native_mean_port_keeps_historical_default_and_signs():
    source = (ROOT/"include/pops/numerics/elliptic/nd/general_field_operator.hpp").read_text()
    assert "bool ArithmeticFaces = (CoefficientComponents != Components)" in source
    assert "if constexpr (!ArithmeticFaces)" in source
    assert "harmonic_tensor_face_average" in source
    assert "low = Real(0.5) * coefficient(lower, slot) + Real(0.5) * center" in source
    assert "high = Real(0.5) * center + Real(0.5) * coefficient(upper, slot)" in source


def test_explicit_mean_changes_method_and_cache_without_rewriting_equation(monkeypatch):
    _, _, old_program, old, _ = physical.authored(width=2, order=(1,0))
    method = physical.CellCenteredNonlinearCoupled
    monkeypatch.setattr(physical, "CellCenteredNonlinearCoupled", lambda **kwargs:
                        method(**kwargs, face_policy="Arithmetic@1"))
    case, layout, new_program, new, _ = physical.authored(width=2, order=(1,0))
    assert old.attrs["source_contract"]["field_problem"] == new.attrs["source_contract"]["field_problem"]
    assert old.attrs["source_contract"]["diffusion"] == new.attrs["source_contract"]["diffusion"]
    assert old.attrs["source_contract"]["local_expressions"] == new.attrs["source_contract"]["local_expressions"]
    assert old_program._ir_hash() != new_program._ir_hash()
    assert new.attrs["solve_request"]["realization"] == {"coefficient_face_policy": FACE}
    assert "coefficient_face_policy" in method(finite_difference_step=1e-6,face_policy="Arithmetic@1").options()


def test_registered_problem_refuses_resealed_foreign_mean(monkeypatch):
    case, layout, program, token, _ = authored(monkeypatch, width=2, order=(1,0))
    # Convert every solver-owned diffusion payload to an admissible literal v1
    # while retaining the actual registered physical D(State) and method v2.
    from pops._frozen_data import freeze_containers
    from pops.fields._program_nonlinear_problem import _request_data
    from pops.time.solve_request import SolveUnknown
    coefficients = token.inputs[1]
    expressions = tuple(("literal", {"kind":"integer", "value":"1"}) for _ in range(4))
    attrs = dict(coefficients.attrs)
    attrs.pop("coefficient_face_policy")
    attrs.update(expressions=expressions, field_dependencies=())
    coefficients = program._replace_value(coefficients, attrs=attrs)
    object.__setattr__(coefficients, "inputs", ())
    source = dict(token.attrs["source_contract"])
    source.pop("coefficient_face_policy")
    source["diffusion"] = expressions
    attrs = dict(token.attrs)
    attrs.pop("coefficient_face_policy")
    attrs.update(contract="pops.spatial-field-residual@1", source_contract=freeze_containers(source))
    token = program._replace_value(token, attrs=attrs)
    object.__setattr__(token, "inputs", (token.inputs[0], coefficients, *token.inputs[2:]))
    actual = token.attrs["solve_request"]
    unknown = SolveUnknown(actual["unknowns"][0]["name"], token.inputs[0])
    from pops.time._program.serialization import _json_ready
    request = _request_data(program,token,unknown,_json_ready(actual["physical_problem"]))
    token = program._replace_value(token, attrs={**token.attrs, "solve_request":request})
    validate_nonlinear_field_request(program,token)  # Own hashes are consistent.
    with pytest.raises(ValueError, match="equation_identity_drift"):
        physical.emit(case, layout)
