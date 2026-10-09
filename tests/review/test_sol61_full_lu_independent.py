"""Independent source/standalone-host receipt; no native AMR or MPI execution."""

from pathlib import Path
import subprocess

import pytest
from pops.solvers import Newton
from pops.time._program.spatial_solve import (
    prepare_spatial_newton,
    dense_resource_contract,
    validate_dense_resource_contract,
)

ROOT = Path(__file__).resolve().parents[2]
FROZEN = "6ea298945e6befc0a65143b5d51c506b7131f8b3"


def body(text, signature):
    start = text.index(signature)
    begin = text.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def frozen_seam(path, signature):
    relative = str(path.relative_to(ROOT))
    reference = subprocess.run(
        ["git", "show", FROZEN + ":" + relative], check=True, capture_output=True, text=True
    ).stdout
    actual = body(path.read_text(), signature)
    assert actual == body(reference, signature)
    return actual


@pytest.mark.parametrize("budget", [None, True, False, 0, -1, 1.0, "1024", 2**64])
def test_resource_contract_rejects_nonexact_uint64(budget):
    with pytest.raises((ValueError, TypeError)):
        Newton(right_preconditioner="FullResidualBasisLU@1", max_dense_bytes=budget)
    with pytest.raises((ValueError, TypeError)):
        dense_resource_contract(budget)


def test_resources_are_part_of_realization_identity_and_seven_controls_are_unchanged():
    baseline = Newton()
    first = Newton(right_preconditioner="FullResidualBasisLU@1", max_dense_bytes=1)
    second = Newton(right_preconditioner="FullResidualBasisLU@1", max_dense_bytes=2**64 - 1)
    assert first.numerical_options() == second.numerical_options() == baseline.numerical_options()
    assert len(first.numerical_options()) == 7
    assert prepare_spatial_newton(first).identity != prepare_spatial_newton(second).identity
    valid = dense_resource_contract(1024)
    assert validate_dense_resource_contract(valid) == 1024
    for poison in (
        {**valid, "version": True},
        {**valid, "scope": "global"},
        {**valid, "extra": 0},
        {**valid, "max_dense_bytes": 1.5},
    ):
        with pytest.raises((ValueError, TypeError)):
            validate_dense_resource_contract(poison)
    with pytest.raises(ValueError):
        Newton(max_dense_bytes=1)
    with pytest.raises(ValueError):
        Newton(right_preconditioner="SpatialBasisJacobi@1", max_dense_bytes=1)
    for encoding in ("FFFFFFFFFFFFFFFF", "0000000000000000", "f" * 17, "1", True):
        with pytest.raises((ValueError, TypeError)):
            validate_dense_resource_contract({**valid, "max_dense_bytes": {"uint64_hex": encoding}})
    for budget in (1, 2**63 - 1, 2**63, 2**64 - 1):
        assert validate_dense_resource_contract(dense_resource_contract(budget)) == budget


@pytest.mark.parametrize("width,order", [(2, (1, 0)), (3, (2, 0, 1))])
def test_public_source_admission_preserves_original_equation_and_emits_uint64(width, order):
    from pops.amr import (
        AMRExecution,
        AMRHierarchy,
        AMRRegrid,
        AMRTagging,
        AMRTransfer,
        Buffer,
        Tag,
        Hysteresis,
        EqualityPolicy,
        ConflictPolicy,
    )
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.math import ValueExpr
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.params import RuntimeParam
    from pops.time import every
    from tests.python.unit.fields.test_nonlinear_mixed_field_problem import mixed_case, finish

    def emit(solver):
        args = mixed_case(order, width=width, solver=solver)
        case, field, program, current, request, block, forcing, frame = args
        transfer = AMRTransfer()
        transfer.state(block[forcing], StateTransfer())
        threshold = case.param(RuntimeParam("threshold", default=0.5))
        layout = AMR(
            grid=CartesianGrid(frame=frame, cells=(16, 12), periodic=PeriodicAxes(frame.axes)),
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(
                rules=(
                    Tag(ValueExpr(block[forcing])["f0"] > case.value(threshold)),
                    Buffer(cells=1),
                ),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                conflict_policy=ConflictPolicy.REFINE_WINS,
            ),
            regrid=AMRRegrid(schedule=every(1000, clock=program.clock)),
            transfer=transfer,
            execution=AMRExecution.synchronous(),
        )
        cpp = finish(*args, layout=layout, target="amr_system")
        token = next(v for v in program._values if v.op == "solve_spatial_field")
        return program, token, cpp

    old, baseline, _ = emit(Newton(tolerance=1e-11))
    program, selected, cpp = emit(
        Newton(
            tolerance=1e-11, right_preconditioner="FullResidualBasisLU@1", max_dense_bytes=2**64 - 1
        )
    )
    assert program._serialize()["version"] == 13
    assert old._serialize()["version"] == 8
    assert selected.attrs["source_contract"] == baseline.attrs["source_contract"]
    assert selected.attrs["local_expressions"] == baseline.attrs["local_expressions"]
    assert selected.attrs["newton_controls"] == baseline.attrs["newton_controls"]
    assert (
        selected.attrs["solve_request"]["equation_identity"]
        == baseline.attrs["solve_request"]["equation_identity"]
    )
    assert "kFullResidualBasisLU" in cpp and "std::uint64_t{18446744073709551615ULL}" in cpp
    assert "original_amr_field_residual" in cpp and "nonfinite_original_field_residual" in cpp
    from pops.codegen.scratch_plan import build_scratch_plan
    import json

    scratch = json.loads(build_scratch_plan(program).to_json())
    carrier = next(row for row in scratch["persistent"] if "full_residual_basis_lu" in row)
    resources = carrier["full_residual_basis_lu"]["resources"]
    assert validate_dense_resource_contract(resources) == 2**64 - 1


def test_uniform_selected_provider_refuses_explicitly_before_emission():
    from tests.python.unit.fields.test_nonlinear_mixed_field_problem import mixed_case, finish

    args = mixed_case(
        solver=Newton(right_preconditioner="FullResidualBasisLU@1", max_dense_bytes=1024)
    )
    with pytest.raises(ValueError, match="FullResidualBasisLU@1.*Uniform is unsupported"):
        finish(*args)


@pytest.mark.parametrize("release", [False, True])
def test_actual_dense_header_permutations_overflow_and_nonfinite(tmp_path, release):
    executable = tmp_path / "dense-lu-host"
    command = ["c++", "-std=c++20", "-O0", "-I" + str(ROOT / "include")]
    if release:
        command.append("-DNDEBUG")
    command += [
        str(Path(__file__).with_name("sol61_full_lu_independent_host.cpp")),
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    receipt = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=10
    )
    assert receipt.stdout == "independent actual-header host checks=385\n"


def test_active_quotient_and_budget_precede_new_allocations():
    path = ROOT / "include/pops/runtime/program/prepared_amr_field_residual.hpp"
    visit = frozen_seam(path, "void visit_active_cells_")
    assert "value >= Real(.5)" in visit and "std::isfinite(value)" in visit
    assert "owners != Real(1)" in visit and "active != Real(0) && active != Real(1)" in visit
    assert visit.index("owners = all_reduce_sum") < visit.index("visit(level, global")
    contribute = frozen_seam(path, "bool contributes_")
    assert "field.contains_local(global)" in contribute
    assert "!field.distribution().replicated() || lane.rank() == 0" in contribute
    prepare = frozen_seam(path, "void prepare_full_lu_")
    assert prepare.index("required > max_dense_bytes_") < prepare.index("lu_.allocate(count)")
    assert prepare.index("required > max_dense_bytes_") < prepare.index("lu_basis_ = allocate_")
    assert "3 * sizeof(Real)" in prepare
    assert "copy_(q, lu_iterate_)" in prepare
    assert "jvp(lu_iterate_, lu_basis_, lu_image_, evaluation)" in prepare
    assert "apply_original_field_operator" not in prepare
    apply = frozen_seam(path, "void apply_full_lu_")
    assert apply.index("visit_active_cells_") < apply.index("gather_active_(input")
    assert apply.index("gather_active_(input") < apply.index("lu_.apply(")


def test_full_original_JVP_and_terminal_residual_are_preserved():
    path = ROOT / "include/pops/runtime/program/prepared_amr_field_residual.hpp"
    text = path.read_text()
    derivative = body(text, "auto derivative =")
    assert derivative.count("evaluate(perturbed_") == 2
    assert "h, direction[level]" in derivative and "-h, direction[level]" in derivative
    assert "Real(0.5) / h, plus_[level], -Real(0.5) / h, minus_[level]" in derivative
    evaluate = body(text, "auto evaluate =")
    assert evaluate.index("synchronize_original_field_candidate") < evaluate.index(
        "coefficient_body("
    )
    assert evaluate.index("coefficient_body(") < evaluate.index(
        "apply_original_field_operator(evaluation_q_"
    )
    assert "add_local(*physical_q, captures_, result, evaluation)" in evaluate
    assert "evaluate(candidate_, recheck_, 0)" in text
    assert "original_amr_field_residual_recheck_failed" in text
    workspace = ROOT / "include/pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp"
    cycle = frozen_seam(workspace, "SolveReport solve_rebuilt_preconditioned")
    assert cycle.count("prepare_right(iterate_, jvp_provider, iteration)") == 1
    assert cycle.index("prepare_right(") < cycle.index("solve_linear_(")


def test_legacy_jacobi_preparation_body_is_exactly_the_parent():
    path = ROOT / "include/pops/runtime/program/prepared_amr_field_residual.hpp"
    reference = subprocess.run(
        ["git", "show", "6596a1749c98861ef87f4b469a39d94e428c40bc:" + str(path.relative_to(ROOT))],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert body(path.read_text(), "void prepare_spatial_jacobi_") == body(
        reference, "void prepare_spatial_jacobi_"
    )


def test_actual_active_scan_with_two_host_rank_adapters(tmp_path):
    """Run the pinned scan itself; adapters implement votes, not native AMR/MPI."""
    path = ROOT / "include/pops/runtime/program/prepared_amr_field_residual.hpp"
    visit = frozen_seam(path, "void visit_active_cells_")
    contribution = frozen_seam(path, "bool contributes_")
    source = (
        r"""
#include <pops/core/foundation/types.hpp>
#include <algorithm>
#include <array>
#include <barrier>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
using pops::Real;
#define POPS_HD
template<int Dim> using Index=std::array<int,Dim>;
template<int Dim> struct Box {
  Index<Dim> lo,hi;
  Box(Index<Dim> a,Index<Dim> b):lo(a),hi(b){}
  int length(int axis)const{return hi[axis]-lo[axis]+1;}
  int numPts()const{return length(0);}
  bool operator==(const Box&)const=default;
};
template<int Dim,class F> Real for_each_cell_reduce_max(Box<Dim> b,F f){return f(b.lo);}
struct World {
  std::barrier<> barrier{2}; std::array<Real,2> slots{};
  Real sum(int rank,Real value){slots[rank]=value;barrier.arrive_and_wait();
    Real result=slots[0]+slots[1];barrier.arrive_and_wait();return result;}
};
struct ExecutionLane {World* world;int id;int rank()const{return id;}};
Real all_reduce_sum(Real value,const ExecutionLane& lane){return lane.world->sum(lane.id,value);}
struct Distribution {bool replicate;bool replicated()const{return replicate;}
  bool operator==(const Distribution&)const=default;};
struct Field {
  std::vector<Box<1>> boxes;std::vector<bool> owns;std::vector<Real> values;
  int rank,width;Distribution dist;
  const auto& layout()const{return boxes;} Distribution distribution()const{return dist;}
  int local_rank()const{return rank;}int ncomp()const{return width;}
  bool contains_local(std::size_t global)const{return owns[global];}
  struct View {const Field* field;Real operator()(Index<1> cell,int)const{
    return field->values.at(cell[0]);}};
  struct Fab {const Field* field;View view()const{return {field};}};
  Fab fab_global(std::size_t)const{return {this};}
};
struct Op {std::vector<Field> masks;const Field& original_field_active_cells(int level)const{
  return masks.at(level);}};
template<int Dim> struct Probe {
  using field_type=Field;std::vector<Field> candidate_;Op* op_;
  template<class Operation> static void local_phase_(const ExecutionLane& lane,Operation operation){
    bool failed=false;try{operation();}catch(...){failed=true;}
    if(all_reduce_sum(failed?1:0,lane)!=0)throw std::invalid_argument("host collective refusal");}
"""
        + contribution
        + "\n template<class Visitor> "
        + visit
        + r"""
};
void require(bool value){if(!value)throw std::runtime_error("active scan assertion failed");}
void run(int mode,bool refusal){
  World world;std::array<bool,2> failed{};std::array<int,2> count{};
  auto rank=[&](int id){
    const bool replicated=mode==2;
    Field coarse{{{{0},{1}},{{2},{3}}},{id==0,id==1},{1,0,1,0},id,2,{replicated}};
    Field fine{{{{4},{5}}},{id==0},{0,0,0,0,1,1},id,2,{replicated}};
    if(mode==1){coarse.owns={id==0,id==0};fine.owns={id==0};} // rank 1 empty
    if(mode==2){coarse.owns={true,true};fine.owns={true};}
    if(mode==3){coarse.owns[0]=true;} // duplicate distributed owner
    if(mode==4){coarse.owns[1]=false;} // missing owner
    if(mode==5 && id==1)coarse.values[2]=std::numeric_limits<Real>::quiet_NaN();
    Op op{{coarse,fine}};for(auto& mask:op.masks)mask.width=1;
    Probe<1> probe{{coarse,fine},&op};ExecutionLane lane{&world,id};
    try{probe.visit_active_cells_(lane,[&](auto,auto,const auto& cell,int width){
      require(cell[0]!=1 && cell[0]!=3);count[id]+=width;});}
    catch(const std::invalid_argument&){failed[id]=true;}
  };
  std::thread a(rank,0),b(rank,1);a.join();b.join();
  require(failed[0]==refusal && failed[1]==refusal);
  if(!refusal)require(count[0]==8 && count[1]==8);
}
int main(){run(0,false);run(1,false);run(2,false);run(3,true);run(4,true);run(5,true);
 std::cout<<"actual active scan host cases=6\n";}
"""
    )
    cpp = tmp_path / "active-scan.cpp"
    cpp.write_text(source)
    executable = tmp_path / "active-scan"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O0",
            "-pthread",
            "-I" + str(ROOT / "include"),
            str(cpp),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    receipt = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=10
    )
    assert receipt.stdout == "actual active scan host cases=6\n"
