"""Source admission and actual small host LU; no native AMR qualification."""
import json
import copy
import subprocess
from pathlib import Path
import pytest
from pops.solvers import Newton
from pops.time import SolveRequestError
from pops.time._program.serialization import _json_ready
from pops.time._program.spatial_solve import (
    FULL_RESIDUAL_BASIS_LU, dense_resource_contract, validate_dense_resource_contract,
)
from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
import test_sol61_amr_public_original as physical
import test_sol61_candidate_diffusion as candidate
import test_sol61_original_captured_diffusion as captured

ROOT = Path(__file__).resolve().parents[2]
BUDGET = 64 * 1024 * 1024


def authored(monkeypatch, *, uniform=False, width=3, candidate_d=False, captured_d=False):
    monkeypatch.setattr(physical, "Newton", lambda **controls:
        Newton(**controls, right_preconditioner="FullResidualBasisLU@1", max_dense_bytes=BUDGET))
    if candidate_d:
        return candidate.authored(monkeypatch, uniform=uniform, width=width)
    if captured_d:
        return captured.authored(monkeypatch, uniform=uniform, width=width)
    return physical.authored(width, uniform=uniform)


@pytest.mark.parametrize("candidate_d,captured_d,width", [(False,False,1), (False,False,3), (False,True,3), (True,False,3)])
def test_real_source_routes_full_residual_lu_and_authenticates_budget(monkeypatch, candidate_d, captured_d, width):
    case, layout, program, token, _ = authored(monkeypatch, width=width,
        candidate_d=candidate_d, captured_d=captured_d)
    validate_nonlinear_field_request(program, token)
    assert token.attrs["right_preconditioner"] == FULL_RESIDUAL_BASIS_LU
    resources = token.attrs["right_preconditioner_resources"]
    assert validate_dense_resource_contract(resources) == BUDGET
    request = _json_ready(token.attrs["solve_request"])
    assert request["realization"]["right_preconditioner_resources"] == dict(resources)
    assert program._serialize()["version"] == 13
    code, resolved = physical.emit(case, layout)
    assert resolved.time._serialize()["version"] == 13
    assert "AmrFieldRightPreconditioner::kFullResidualBasisLU" in code
    assert f"std::uint64_t{{{BUDGET}ULL}}" in code
    assert "kPerCandidate" in code if candidate_d else "AmrFieldCoefficientEvaluation::kFrozen" in code
    from pops.codegen.scratch_plan import build_scratch_plan
    plan = json.dumps(build_scratch_plan(resolved.time).to_dict())
    assert "2*Nactive" in plan and "O(Nactive**3)" in plan
    assert "replicated_per_rank_active_owned_quotient" in plan


def test_uniform_refuses_only_this_unported_realization(monkeypatch):
    case, layout, _, _, _ = authored(monkeypatch, uniform=True)
    with pytest.raises((ValueError, NotImplementedError), match="FullResidualBasisLU@1.*Uniform is unsupported"):
        physical.emit(case, layout)


@pytest.mark.parametrize("budget", [None, 0, -1, True, 1.0, "1000", 2**64])
def test_resource_budget_requires_exact_positive_uint64(budget):
    with pytest.raises(ValueError, match="max_dense_bytes"):
        Newton(right_preconditioner="FullResidualBasisLU@1", max_dense_bytes=budget)


@pytest.mark.parametrize("policy", [None, "SpatialBasisJacobi@1"])
def test_dense_budget_cannot_silently_change_legacy_realization(policy):
    with pytest.raises(ValueError, match="belongs only"):
        Newton(right_preconditioner=policy, max_dense_bytes=BUDGET)


def test_prepared_mutation_and_resource_scope_refused():
    solver = Newton(right_preconditioner="FullResidualBasisLU@1", max_dense_bytes=BUDGET)
    prepared = solver.prepare_program_solve()
    object.__setattr__(prepared, "max_dense_bytes", BUDGET+1)
    with pytest.raises(SolveRequestError, match="realization changed"):
        prepared.__post_init__()
    good = dense_resource_contract(BUDGET)
    assert validate_dense_resource_contract(good) == BUDGET
    for data in ({**good, "version":True}, {**good, "version":1.0}, {**good, "scope":"global"}, {**good,"extra":1}):
        with pytest.raises(SolveRequestError):
            validate_dense_resource_contract(data)



def test_resource_choice_preserves_numerical_controls_and_deep_copy():
    legacy = Newton(tolerance=2e-9, max_iterations=12, linear_tolerance=1e-5,
                    linear_max_iterations=240, restart=80, minimum_step=1e-6)
    selected = Newton(**legacy.numerical_options(), right_preconditioner="FullResidualBasisLU@1",
                      max_dense_bytes=2**64-1)
    assert selected.numerical_options() == legacy.numerical_options()
    retained = copy.deepcopy(selected).prepare_program_solve()
    retained.__post_init__()
    assert retained.max_dense_bytes == 2**64-1
    assert retained.right_preconditioner == FULL_RESIDUAL_BASIS_LU
    assert validate_dense_resource_contract(dense_resource_contract(retained.max_dense_bytes)) == 2**64-1


@pytest.mark.parametrize("hexadecimal", ["0", "0000000000000000", "FFFFFFFFFFFFFFFF", "000000000000000g", True])
def test_unsigned_budget_encoding_is_canonical_and_positive(hexadecimal):
    good = dense_resource_contract(BUDGET)
    assert validate_dense_resource_contract(good) == BUDGET
    with pytest.raises(SolveRequestError):
        validate_dense_resource_contract({**good, "max_dense_bytes":{"uint64_hex":hexadecimal}})


def test_equal_string_subclass_is_not_a_typed_realization():
    class Foreign(str):
        pass
    solver = Newton(right_preconditioner="FullResidualBasisLU@1", max_dense_bytes=BUDGET)
    solver._right_preconditioner = Foreign("FullResidualBasisLU@1")
    with pytest.raises(SolveRequestError, match="unknown Newton"):
        solver.prepare_program_solve()


def test_resealed_budget_drift_cannot_reuse_solver_authority(monkeypatch):
    _, _, program, token, _ = authored(monkeypatch)
    resources = dict(token.attrs["right_preconditioner_resources"])
    resources["max_dense_bytes"] = {"uint64_hex":f"{BUDGET+1:016x}"}
    token = program._replace_value(token, attrs={**token.attrs, "right_preconditioner_resources":resources})
    # The physical request mirror must agree as well; resealing it still cannot
    # preserve the original solver identity with a different resource budget.
    request = _json_ready(token.attrs["solve_request"])
    request["realization"]["right_preconditioner_resources"] = resources
    token = program._replace_value(token, attrs={**token.attrs, "solve_request":request})
    with pytest.raises(SolveRequestError, match="Newton controls changed"):
        validate_nonlinear_field_request(program, token)


def test_source_uses_complete_jvp_rebuild_and_real_active_owned_quotient():
    source = (ROOT/"include/pops/runtime/program/prepared_amr_field_residual.hpp").read_text()
    assert "jvp(lu_iterate_, lu_basis_, lu_image_, evaluation)" in source
    assert "original_field_active_cells" in source
    assert "owners != Real(1)" in source and "!field.distribution().replicated() || lane.rank() == 0" in source
    assert source.index("required > max_dense_bytes_") < source.index("lu_.allocate(count)")
    assert "active quotient changed" in source
    workspace = (ROOT/"include/pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp").read_text()
    assert workspace.index("prepare_right(iterate_, jvp_provider, iteration)") < workspace.index("solve_linear_(iterate_, residual_")
    assert "full_correction" in workspace or "apply_jvp(iterate, correction_" in workspace
    assert "full_residual_basis_lu.hpp" in (ROOT/"include/pops_headers.manifest").read_text()


def test_actual_host_lu_pivots_signed_matrix_and_refuses_singular_or_overflow(tmp_path):
    # Compile the unchanged numeric class, replacing only its Real type include.
    # No Kokkos/MPI/native runtime, DSO, or scientific solver is executed here.
    source = (ROOT/"include/pops/numerics/elliptic/interface/full_residual_basis_lu.hpp").read_text()
    source = source.replace("#pragma once", "").replace("#include <pops/core/foundation/types.hpp>", "namespace pops { using Real = double; }")
    source += r'''
#include <array>
#include <cassert>
int main() {
  pops::FullResidualBasisLU lu;
  lu.allocate(3);
  // Multiple row pivots, nonsymmetric, negative diagonal, no SPD assumption.
  const std::array<double,9> a{0,2,-1, 1,-3,2, 7,-20,0};
  for(std::size_t i=0;i<3;++i) for(std::size_t j=0;j<3;++j) lu.entry(i,j)=a[3*i+j];
  lu.factor();
  const std::array<double,3> expected{.25,-.5,1.25};
  std::array<double,3> rhs{}, result{};
  for(std::size_t i=0;i<3;++i) for(std::size_t j=0;j<3;++j) rhs[i]+=a[3*i+j]*expected[j];
  lu.apply(rhs,result);
  for(std::size_t i=0;i<3;++i) assert(std::abs(result[i]-expected[i])<1e-14);
  assert(pops::FullResidualBasisLU::required_bytes(3,17)==17+9*sizeof(double)+3*(2*sizeof(double)+sizeof(std::size_t)));
  bool overflow=false;try { pops::FullResidualBasisLU::required_bytes(std::numeric_limits<std::size_t>::max()); } catch(const std::length_error&) { overflow=true; } assert(overflow);
  lu.allocate(2);lu.entry(0,0)=1;lu.entry(0,1)=2;lu.entry(1,0)=2;lu.entry(1,1)=4;
  bool singular=false;try { lu.factor(); } catch(const std::invalid_argument&) { singular=true; } assert(singular);
  lu.entry(1,1)=std::numeric_limits<double>::infinity();
  bool nonfinite=false;try { lu.factor(); } catch(const std::invalid_argument&) { nonfinite=true; } assert(nonfinite);
}
'''
    path = tmp_path/"actual_lu.cpp"
    path.write_text(source)
    binary = tmp_path/"actual_lu"
    subprocess.run(["/usr/bin/clang++","-std=c++20","-O0",str(path),"-o",str(binary)],check=True,timeout=30,capture_output=True)
    subprocess.run([str(binary)],check=True,timeout=10,capture_output=True)
