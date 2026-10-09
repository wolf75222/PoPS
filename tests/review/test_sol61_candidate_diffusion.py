"""Full candidate-dependent physical diffusion: source admission and exact math only."""
import json
from fractions import Fraction
from pathlib import Path
import pytest
from pops.fields import CellCenteredNonlinearCoupled
from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
from pops.math import ValueExpr
from pops.time import SolveRequestError
from pops.time._program.serialization import _json_ready
import test_sol61_amr_public_original as physical
import sol61_candidate_diffusion_counterexample as exact

POLICY = "pops.field.coefficients.per-candidate@1"


def authored(monkeypatch, *, uniform=False, width=3, order=None):
    states = []
    declare = physical.pops.Model.state
    def state(model, *args, **kwargs):
        value = declare(model, *args, **kwargs)
        states.append(value)
        return value
    monkeypatch.setattr(physical.pops.Model, "state", state)
    div = physical.DivCoeffGrad
    monkeypatch.setattr(physical, "DivCoeffGrad", lambda field, coefficient:
        div(field, coefficient * (1 + states[-1][0] + ValueExpr(field)**2)))
    method = physical.CellCenteredNonlinearCoupled
    monkeypatch.setattr(physical, "CellCenteredNonlinearCoupled", lambda **kwargs:
        method(**kwargs, face_policy="Arithmetic@1", coefficient_evaluation="PerCandidate@1"))
    return physical.authored(width, order, uniform=uniform)


@pytest.mark.parametrize("uniform,width,order", [(True,1,(0,)), (True,3,(2,0,1)), (False,3,(2,0,1)), (False,5,(4,1,3,0,2))])
def test_true_candidate_body_reaches_full_native_residual(monkeypatch, uniform, width, order):
    case, layout, program, token, _ = authored(monkeypatch, uniform=uniform, width=width, order=order)
    validate_nonlinear_field_request(program, token)
    descriptor = token.inputs[1]
    assert descriptor.attrs["storage_role"] == "deferred_candidate_coefficient"
    assert descriptor.attrs["expressions"] == token.attrs["source_contract"]["diffusion"]
    assert "unknown" in json.dumps(_json_ready(descriptor.attrs["expressions"]))
    assert descriptor.inputs == token.inputs[2:2+token.attrs["capture_count"]]
    assert token.attrs["contract"] == "pops.spatial-field-residual@3"
    assert token.attrs["coefficient_evaluation"] == POLICY
    assert token.attrs["linear_residual_verification"] == "pops.field.linear.true-correction-residual@1"
    assert token.attrs["solve_request"]["realization"]["linear_residual_verification"] == token.attrs["linear_residual_verification"]
    assert token.attrs["solve_request"]["equation_inputs"]["diffusion_body"] == descriptor.attrs["expressions"]
    assert program._serialize()["version"] == 11
    code, resolved = physical.emit(case, layout)
    assert resolved.time._serialize()["version"] == 11
    from pops.codegen.scratch_plan import build_scratch_plan
    assert "no second GMRES basis" in json.dumps(build_scratch_plan(resolved.time).to_dict())
    if uniform:
        callback = code.index("_evaluate =")
        assert "prepare_general_field_coefficients" not in code[:callback]
        assert code.index("nonfinite_candidate_diffusion", callback) < code.index("prepare_general_field_coefficients", callback)
        assert f"apply_general_field<pops::kNativeDimension, {width}, {width*width}, true, true>" in code
        assert "same_attempt(ctx.resource_attempt())" in code
        assert "candidate coefficient point/attempt/lane authority changed" in code
        assert "coefficient(index," in code[callback:]
    else:
        assert code.count("_core->solve_candidate(") == 1
        assert "AmrFieldCoefficientEvaluation::kPerCandidate" in code
        assert "coefficients[level]->box(patch)" in code
        assert "q[level].fab(patch).view()" in code
    assert "nonfinite_candidate_diffusion" in code
    assert "nonfinite_original_field_residual" in code
    # Evaluate the encoded physical D using the unknown product's actual permutation.
    captures = token.inputs[2:2+token.attrs["capture_count"]]
    alpha = next(i for i, value in enumerate(captures) if tuple(value.space.components) == ("alpha",))
    inputs = [[Fraction(-3,2)] if i == alpha else [Fraction(0)]*len(value.space.components)
              for i, value in enumerate(captures)]
    q = tuple(Fraction(i+1,7) for i in range(width))
    matrix = physical.matrix(width)
    for i in range(width):
        for j in range(width):
            actual = physical.evaluate(descriptor.attrs["expressions"][i*width+j], q, inputs)
            assert actual == pytest.approx(float(matrix[order[i]][order[j]]*(1+inputs[alpha][0]+q[j]**2)))


@pytest.mark.parametrize("policy", [True, 1, "PerCandidate", "PerCandidate@2", {}, "Frozen@1"])
def test_candidate_realization_is_exact_named_version(policy):
    with pytest.raises(ValueError, match="coefficient_evaluation"):
        CellCenteredNonlinearCoupled(finite_difference_step=1e-6, face_policy="Arithmetic@1", coefficient_evaluation=policy)


def test_candidate_realization_requires_explicit_arithmetic():
    with pytest.raises(ValueError, match="Arithmetic"):
        CellCenteredNonlinearCoupled(finite_difference_step=1e-6, coefficient_evaluation="PerCandidate@1")


def test_frozen_linear_spatial_preconditioner_is_not_relabelled_nonlinear(monkeypatch):
    from pops.solvers import Newton
    monkeypatch.setattr(physical, "Newton", lambda **controls:
                        Newton(**controls, right_preconditioner="SpatialBasisJacobi@1"))
    with pytest.raises(SolveRequestError, match="frozen linear spatial operator"):
        authored(monkeypatch)


@pytest.mark.parametrize("mutation", ["unknown", "capture", "point", "role", "policy"])
def test_deferred_foreign_descriptor_refused(monkeypatch, mutation):
    _, _, program, token, _ = authored(monkeypatch)
    descriptor = token.inputs[1]
    attrs = dict(descriptor.attrs)
    kwargs = {}
    if mutation == "unknown":
        attrs["unknown_components"] = tuple(reversed(attrs["unknown_components"]))
    elif mutation == "capture":
        kwargs["inputs"] = tuple(reversed(descriptor.inputs))
    elif mutation == "point":
        kwargs["point"] = None
    elif mutation == "role":
        attrs["storage_role"] = "materialized"
    else:
        attrs["coefficient_evaluation"] = "pops.field.coefficients.per-candidate@2"
    descriptor = program._replace_value(descriptor, attrs=attrs)
    for key,value in kwargs.items():
        object.__setattr__(descriptor,key,value)
    object.__setattr__(token,"inputs",(token.inputs[0],descriptor,*token.inputs[2:]))
    with pytest.raises((ValueError,SolveRequestError)):
        validate_nonlinear_field_request(program, token)


def test_exact_full_direction_contains_delta_d_grad_q():
    receipt = exact.run()
    assert receipt["scalar"]["original_DF"] != receipt["scalar"]["frozen_at_q_DF"]


def test_native_full_composition_and_private_apply_resource_source():
    root = Path(__file__).resolve().parents[2]
    text = (root/"include/pops/runtime/program/prepared_amr_field_residual.hpp").read_text()
    start = text.index("auto evaluate =")
    end = text.index("auto residual =", start)
    evaluate = text[start:end]
    order = ["copy_(q, evaluation_q_)", "synchronize_original_field_candidate(evaluation_q_)",
             "coefficient_body(", "prepare_original_field_operator()", "apply_original_field_operator(evaluation_q_", "add_local(*physical_q"]
    offsets = [evaluate.index(value) for value in order]
    assert offsets == sorted(offsets)
    assert "evaluate(perturbed_, plus_" in text and "evaluate(perturbed_, minus_" in text
    assert "evaluate(candidate_, recheck_" in text
    assert "evaluation_generation_" in text
    assert "authenticate_coefficients_();" in evaluate
    provider = (root/"include/pops/numerics/elliptic/nd/prepared_composite_general_field.hpp").read_text()
    assert "if (!apply_only_)" in provider
    assert "candidate coefficient resource is apply-only" in provider


def test_registered_physics_refuses_consistently_resealed_foreign_d(monkeypatch):
    from pops._frozen_data import freeze_containers
    from pops.fields._program_nonlinear_problem import _request_data
    from pops.time.solve_request import SolveUnknown
    case, layout, program, token, _ = authored(monkeypatch)
    coefficient = token.inputs[1]
    # Modify the physical D body coherently in every solver-owned location.
    literal = ("literal", {"kind":"integer", "value":"2"})
    diffusion = tuple(("mul", expression, literal) for expression in coefficient.attrs["expressions"])
    coefficient = program._replace_value(coefficient, attrs={**coefficient.attrs, "expressions":diffusion})
    source = dict(token.attrs["source_contract"])
    source["diffusion"] = diffusion
    token = program._replace_value(token, attrs={**token.attrs, "source_contract":freeze_containers(source)})
    object.__setattr__(token, "inputs", (token.inputs[0],coefficient,*token.inputs[2:]))
    actual = token.attrs["solve_request"]
    unknown = SolveUnknown(actual["unknowns"][0]["name"], token.inputs[0])
    request = _request_data(program,token,unknown,_json_ready(actual["physical_problem"]))
    token = program._replace_value(token, attrs={**token.attrs, "solve_request":request})
    validate_nonlinear_field_request(program,token)
    with pytest.raises(ValueError, match="equation_identity_drift"):
        physical.emit(case, layout)


def test_deferred_body_cannot_escape_as_material_field(monkeypatch):
    _, _, program, token, _ = authored(monkeypatch)
    descriptor = token.inputs[1]
    program._new("scalar_field","linear_combine",(descriptor,),{"coeffs":(1,)},"foreign-material-D",None,inherit_state_ref=False)
    with pytest.raises(SolveRequestError, match="body descriptor"):
        validate_nonlinear_field_request(program,token)


def test_uniform_guarded_native_phases_do_not_change_defaults():
    root = Path(__file__).resolve().parents[2]
    workspace = (root/"include/pops/numerics/elliptic/interface/field_newton_krylov.hpp").read_text()
    spatial = (root/"include/pops/runtime/program/prepared_spatial_residual.hpp").read_text()
    assert "bool guard_local = false" in workspace and "bool guard_local = false" in spatial
    assert "const auto& prepared_lane = guard_local_ ? *authority_lane_ : lane" in spatial
    assert "&lane != authority_lane_" in spatial
    assert "local_phase_(prepared_lane" in spatial
    for local_operation in ("product = dot_all_local", "saxpy(work_", "updated = update_correction_", "local_squared = dot_all_local"):
        assert "local_phase_(lane, [&] { " + local_operation in workspace
    assert "newton_.solve(candidate_, defect, derivative, no_gauge, prepared_lane, guard_local_, verify_true_correction_)" in spatial
    assert "bool verify_true_correction = false" in workspace and "bool verify_true_correction = false" in spatial
    assert "if (cycle_converged && !verify_true_correction_)" in workspace
    assert workspace.index("apply_jvp(iterate, correction_, image_") < workspace.index("beta = norm_(linear_residual_, lane)", workspace.index("apply_jvp(iterate, correction_, image_"))


def test_exact_native_local_phase_fault_schedule(tmp_path):
    # Extract the unchanged real helper, with only Kokkos/lane/vote host seams.
    # This proves exception scheduling, not native MPI execution or a PDE runtime.
    import subprocess
    root = Path(__file__).resolve().parents[2]
    header = (root/"include/pops/numerics/elliptic/nd/general_field_operator.hpp").read_text()
    start = header.index("template <bool Guarded, class Operation>")
    end = header.index("/// The scalar Program", start)
    method = header[start:end]
    source = r"""#include <exception>
#include <stdexcept>
#include <string>
#include <iostream>
struct ExecutionLane { int id; };
std::string events;
bool fence_fail=false;
namespace Kokkos { void fence() { events+='F'; if(fence_fail) throw std::runtime_error("fence"); } }
void collectively_rethrow_exception(std::exception_ptr error, const ExecutionLane& lane, const char*) {
  if(lane.id!=7) throw std::logic_error("foreign vote lane"); events+='V'; if(error) std::rethrow_exception(error);
}
""" + method + r"""
int main() {
  ExecutionLane lane{7};
  for(int mode=0;mode<4;++mode) {
    events.clear();fence_fail=mode==2;bool refused=false;
    try {
      auto work=[&] { events+='L'; if(mode==1) throw std::runtime_error("launch"); };
      if(mode==3) general_field_local_phase<false>(lane,work);
      else general_field_local_phase<true>(lane,work);
    } catch(...) { refused=true; }
    if(events!=(mode==3?"L":"LFV") || refused!=(mode==1||mode==2)) return 2;
  }
  std::cout<<"4 real-helper schedules PASS\n";
}
"""
    cpp = tmp_path/"guard.cpp"
    cpp.write_text(source)
    binary = tmp_path/"guard"
    subprocess.run(["/usr/bin/clang++","-std=c++20",str(cpp),"-o",str(binary)],check=True,timeout=20,capture_output=True)
    result = subprocess.run([str(binary)],check=True,timeout=5,capture_output=True,text=True)
    assert "4 real-helper schedules PASS" in result.stdout


def legacy_images():
    """Fresh old-policy images at a fixed fixture/callsite, for separate source roots."""
    import hashlib
    import test_sol61_original_captured_diffusion as captured
    from pops.solvers import Newton
    result = {}
    for name, uniform, has_capture, jacobi in (
            ("legacy",True,False,False), ("captured",True,True,False),
            ("jacobi",False,False,True), ("captured-amr-jacobi",False,True,True)):
        with pytest.MonkeyPatch.context() as patch:
            if jacobi:
                patch.setattr(physical,"Newton",lambda **controls: Newton(**controls,right_preconditioner="SpatialBasisJacobi@1"))
            case,layout,_,token,_ = (captured.authored(patch,uniform=uniform,width=3,order=(2,0,1))
                if has_capture else physical.authored(3,(2,0,1),uniform=uniform))
            code,resolved = physical.emit(case,layout)
            result[name] = {"ir":resolved.time._ir_hash(), "ir_full":resolved.time._serialize(),
                "cpp":hashlib.sha256(code.encode()).hexdigest(),
                "modules":[block.model.module.module_hash() for block in resolved.blocks],
                "manifests":[block.model.module.manifest().to_dict() for block in resolved.blocks],
                "request":_json_ready(token.attrs["solve_request"])}
    return result


@pytest.mark.parametrize("criterion", [None, True, "TrueCorrectionResidual@2", "pops.field.linear.projected-residual@1"])
def test_candidate_true_linear_criterion_cannot_be_removed_or_relabelled(monkeypatch, criterion):
    _,_,program,token,_ = authored(monkeypatch)
    token = program._replace_value(token,attrs={**token.attrs,"linear_residual_verification":criterion})
    with pytest.raises(SolveRequestError,match="true correction residual"):
        validate_nonlinear_field_request(program,token)
