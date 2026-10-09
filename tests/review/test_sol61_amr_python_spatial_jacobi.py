"""Public realization authoring/IR/resolve/emission only: no installed native calls."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest
from pops.identity import make_identity
from pops.solvers import Newton
from pops.time import SolveRequestError
from pops.time._program.spatial_solve import SPATIAL_BASIS_JACOBI, spatial_solver_identity
from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
import test_sol61_amr_public_original as physical

ROOT = Path(__file__).resolve().parents[2]


def selected(monkeypatch, **kwargs):
    # Reuse the independently authored physical body; the only change is the
    # declared numerical realization, made before its request/Program exists.
    monkeypatch.setattr(physical, "Newton", lambda **controls:
                        Newton(**controls, right_preconditioner="SpatialBasisJacobi@1"))
    return physical.authored(**kwargs)


@pytest.mark.parametrize("width,order,seed", [(2,(1,0),False),(3,(2,0,1),True),(5,(4,1,3,0,2),True)])
def test_true_original_body_and_explicit_native_policy(monkeypatch, width, order, seed):
    case, layout, program, token, _ = selected(monkeypatch, width=width, order=order, seed=seed)
    code, resolved = physical.emit(case, layout)
    assert "AmrFieldRightPreconditioner::kSpatialBasisJacobi" in code
    assert code.count("_core->solve(") == 1
    assert "original_hierarchy_field_authority" in code
    assert "stage_original_field_candidate_collectively" in code
    assert "nonfinite_original_field_residual" in code
    assert "response_source" in code
    assert token.attrs["solve_request"]["schema_version"] == 2
    assert token.attrs["solve_request"]["realization"] == {"right_preconditioner": SPATIAL_BASIS_JACOBI}
    assert program._serialize()["version"] == 9
    assert resolved.time._serialize()["version"] == 9
    assert token.attrs["solver_identity"] == spatial_solver_identity(token.attrs["newton_controls"], SPATIAL_BASIS_JACOBI).token
    from pops.codegen.scratch_plan import build_scratch_plan
    budget = build_scratch_plan(resolved.time).to_dict()
    assert "stored DOFs" in json.dumps(budget)


def test_identity_default_exact_legacy_controls_and_policy_absence():
    solver = Newton()
    assert solver.right_preconditioner is None
    assert "right_preconditioner" not in solver.to_data()
    prepared = solver.prepare_program_solve()
    assert prepared.identity == make_identity("prepared-spatial-newton", prepared.controls.to_data())
    case, layout, program, token, _ = physical.authored()
    code, _ = physical.emit(case, layout)
    assert "AmrFieldRightPreconditioner::" not in code
    assert "right_preconditioner" not in token.attrs
    assert token.attrs["solve_request"]["schema_version"] == 1
    assert program._serialize()["version"] == 8


@pytest.mark.parametrize("policy", [True, 1, "SpatialBasisJacobi", "SpatialBasisJacobi@2", "Identity@1", {}])
def test_unknown_authored_realization_refused(policy):
    with pytest.raises(ValueError, match="right_preconditioner"):
        Newton(right_preconditioner=policy)


def test_prepared_policy_is_detached_and_authenticated():
    solver = Newton(right_preconditioner="SpatialBasisJacobi@1")
    prepared = deepcopy(solver).prepare_program_solve()
    solver._right_preconditioner = None
    assert prepared.right_preconditioner == SPATIAL_BASIS_JACOBI
    assert prepared.identity != Newton().prepare_program_solve().identity
    with pytest.raises(SolveRequestError, match="realization"):
        replace(prepared, right_preconditioner=None)
    with pytest.raises(AttributeError):
        solver.right_preconditioner = "SpatialBasisJacobi@1"


def test_uniform_admission_refuses_before_emission(monkeypatch):
    case, layout, _, _, _ = selected(monkeypatch, uniform=True)
    with pytest.raises(ValueError, match="Uniform is unsupported"):
        physical.emit(case, layout)


def test_installed_field_plan_does_not_silently_ignore_policy():
    with pytest.raises(ValueError, match="installed field plans do not implement"):
        Newton(right_preconditioner="SpatialBasisJacobi@1").lower_field_nonlinear(target="amr_system", layout=None)


@pytest.mark.parametrize("mutation", [None, True, "pops.amr.original-spatial-jacobi.basis-response@2", "SpatialBasisJacobi@1"])
def test_resigned_unknown_or_erased_policy_refused(monkeypatch, mutation):
    _, _, program, token, _ = selected(monkeypatch)
    data = dict(token.attrs["solve_request"])
    data["realization"] = {"right_preconditioner": mutation}
    controls = token.attrs["newton_controls"]
    from pops.time._program.serialization import _json_ready
    fake = make_identity("prepared-spatial-newton-v2", {"controls": _json_ready(controls), "right_preconditioner": mutation}).token
    data["solver_identity"] = fake
    token = program._replace_value(token, attrs={**token.attrs, "right_preconditioner": mutation,
                                               "solver_identity": fake, "solve_request": data})
    with pytest.raises(SolveRequestError):
        validate_nonlinear_field_request(program, token)


def test_policy_changes_compiled_ir_but_preserves_physical_equation(monkeypatch):
    _, _, old_program, old, _ = physical.authored()
    _, _, new_program, new, _ = selected(monkeypatch)
    assert old.attrs["solve_request"]["equation_identity"] == new.attrs["solve_request"]["equation_identity"]
    assert old.attrs["newton_controls"] == new.attrs["newton_controls"]
    assert old.attrs["solver_identity"] != new.attrs["solver_identity"]
    assert old_program._ir_hash() != new_program._ir_hash()


def test_no_physics_or_numeric_controls_changed_in_emission_source():
    path = "python/pops/codegen/program_emit_amr_original_field.py"
    old = subprocess.check_output(["git", "show", "c7cdd2c38cc7145abb5981b90f38e9b6035b6e5e:"+path], cwd=ROOT, text=True)
    new = (ROOT/path).read_text()
    assert new.split("callback = stem",1)[1] == old.split("callback = stem",1)[1]
    assert "linear_max_iterations" in old


def test_public_options_parse_and_case_default_keep_selected_realization(monkeypatch):
    solver = Newton(right_preconditioner="SpatialBasisJacobi@1")
    assert Newton(**solver.options()).prepare_program_solve().identity == solver.prepare_program_solve().identity
    assert solver.to_data()["right_preconditioner"] == "SpatialBasisJacobi@1"
    assert "right_preconditioner" not in solver.numerical_options()
    issued = []
    original = physical.pops.Case.field
    def declare(case, *args, **kwargs):
        field = original(case, *args, **kwargs)
        issued.append(field)
        return field
    monkeypatch.setattr(physical.pops.Case, "field", declare)
    _, _, _, token, _ = selected(monkeypatch, width=2, order=(1,0))
    # Program stores a detached canonical handle; use the actual Case-issued
    # declaration for its default solver authority, preserving that guard.
    default = issued[0].default_program_solver()
    assert default.right_preconditioner == "SpatialBasisJacobi@1"
    assert default.prepare_program_solve().identity.token == token.attrs["solver_identity"]


def test_non_original_adapter_refuses_before_invoking_problem_builder():
    from pops.time.solve_request import SolveRequest
    foreign = object.__new__(SolveRequest)
    object.__setattr__(foreign, "problem", object())
    prepared = Newton(right_preconditioner="SpatialBasisJacobi@1").prepare_program_solve()
    with pytest.raises(SolveRequestError, match="only original AMR"):
        prepared.build_program_solve(program=None, problem=foreign)
