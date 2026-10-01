"""Independent public API probes, to be run on the frozen owner extension.

This file does not import the implementation author's fixtures or qualify native
execution.  Its baseline builder is the previously frozen independent witness.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest

from pops.math import ddt, div
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume


def _baseline(*, nonlinear=True, endpoint=True):
    source = Path(__file__).with_name("sol61_history_storage_witness.py")
    spec = importlib.util.spec_from_file_location("independent_owner_baseline", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.authored(nonlinear=nonlinear, endpoint=endpoint)


def _storage_only(width, *, declare=True, foreign_clock=False):
    import pops
    case, program, blocks, states, problem, observation, field, point = _baseline()
    # This fourth block is a real Case member with a declared TimeState, but is
    # absent from the already authored solve's captures and original equations.
    frame = blocks[0]._instance_registry._blocks[blocks[0].local_id]["model"].frame
    model = pops.Model("storage-only-model", frame=frame)
    state = model.state("S", components=tuple("s%d" % i for i in range(width)))
    flux = model.flux("stationary", frame=frame, state=state,
        components={axis: tuple(0*x for x in state) for axis in frame.axes},
        waves={axis: tuple(0*x for x in state) for axis in frame.axes})
    rate = model.rate("stationary-balance", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    owner = case.block("storage-only", model)
    case.numerics(numerics, block=owner)
    from pops.time.points import Clock
    clock = Clock("different-clock", owner=program.owner_path) if foreign_clock else program.clock
    issued = program.state(owner[state], clock=clock) if declare else None
    value = observation[field[problem.unknowns[2]]]
    return case, program, blocks, states, problem, value, field, point, owner, issued


def _authoring_image(program):
    """Include live tables as well as IR, detecting failed-call metadata writes."""
    tables = {}
    for name, value in vars(program).items():
        if isinstance(value, dict):
            tables[name] = tuple((id(k), id(v)) for k, v in value.items())
        elif isinstance(value, list):
            tables[name] = tuple(id(v) for v in value)
        elif isinstance(value, set):
            tables[name] = frozenset(id(v) for v in value)
    return program._serialize(), tables


@pytest.mark.parametrize("width", (3, 5))
def test_storage_only_owner_preserves_global_scalar_and_original_solve(width):
    _case, program, _blocks, _states, problem, value, _field, point, owner, _issued = _storage_only(width)
    solve = value.inputs[0]
    before_inputs = tuple(solve.inputs)
    before_equations = tuple(problem.equations)
    before_attrs = dict(solve.attrs)
    program.store_history("global-component", value, depth=1, owner_block=owner)
    assert value.block is value.state_ref is None and value.point == point
    assert program._histories_ncomp["global-component"] == 1
    assert program._history_blocks["global-component"] is owner
    assert tuple(solve.inputs) == before_inputs
    assert dict(solve.attrs) == before_attrs
    assert all(left is right for left, right in zip(problem.equations, before_equations, strict=True))


def test_foreign_case_owner_refuses_without_authoring_publication():
    _case, program, *_prefix, value, _field, _point, _owner, _issued = _storage_only(3)
    foreign = _storage_only(3)[-2]
    before = _authoring_image(program)
    with pytest.raises((TypeError, ValueError)):
        program.store_history("foreign-storage", value, depth=1, owner_block=foreign)
    assert _authoring_image(program) == before


def test_detached_equal_owner_refuses_without_authoring_publication():
    from pops.problem.handles import BlockHandle
    _case, program, *_prefix, value, _field, _point, owner, _issued = _storage_only(5)
    detached = BlockHandle(owner.local_id, owner=owner.owner_path,
                           model_owner=owner.model_owner_path)
    assert detached == owner and detached is not owner
    before = _authoring_image(program)
    with pytest.raises((TypeError, ValueError)):
        program.store_history("detached-storage", value, depth=1, owner_block=detached)
    assert _authoring_image(program) == before


def test_existing_ring_does_not_accept_another_valid_same_geometry_owner():
    _case, program, blocks, _states, _problem, value, _field, _point, owner, _issued = _storage_only(3)
    program.store_history("fixed-owner", value, depth=1, owner_block=owner)
    before = _authoring_image(program)
    with pytest.raises((TypeError, ValueError)):
        program.store_history("fixed-owner", value, depth=1, owner_block=blocks[0])
    assert _authoring_image(program) == before


@pytest.mark.parametrize("declare,foreign_clock", ((False, False), (True, True)))
def test_missing_or_other_clock_timescope_refuses_atomically(declare, foreign_clock):
    _case, program, *_prefix, value, _field, _point, owner, _issued = _storage_only(
        3, declare=declare, foreign_clock=foreign_clock)
    before = _authoring_image(program)
    with pytest.raises(ValueError, match="TimeState scope|observation clock"):
        program.store_history("unqualified-storage", value, depth=1, owner_block=owner)
    assert _authoring_image(program) == before


def test_observation_alias_does_not_borrow_issued_storage_authority():
    from types import SimpleNamespace
    _case, program, *_prefix, value, _field, _point, owner, _issued = _storage_only(3)
    alias = SimpleNamespace(**{name: getattr(value, name) for name in
        ("op", "vtype", "prog", "inputs", "attrs", "block", "state_ref", "space", "point", "region")})
    before = _authoring_image(program)
    with pytest.raises(ValueError):
        program.store_history("aliased-storage", alias, depth=1, owner_block=owner)
    assert _authoring_image(program) == before


@pytest.mark.parametrize("key,replacement", (
    ("ncomp", 2), ("contract", "pops.program.global-field-history-storage@2"),
    ("representation", "other"), ("sampling", "face"),
    ("owner_block", None), ("storage_state_witness", None), ("layout_witness", None),
    ("field_unknown", None), ("field_problem_identity", None), ("point", None), ("clock", None),
))
def test_individual_storage_descriptor_mutations_are_refused(key, replacement):
    _case, program, *_prefix, value, _field, _point, owner, _issued = _storage_only(3)
    node = program.store_history("descriptor-mutation", value, depth=1, owner_block=owner)
    original = dict(node.attrs["global_field_storage"])
    object.__setattr__(node, "attrs", dict(node.attrs) | {"global_field_storage": original | {key: replacement}})
    with pytest.raises(ValueError):
        program._ir_hash()


@pytest.mark.parametrize("nonlinear,endpoint", ((False, False), (False, True), (True, False), (True, True)))
def test_public_linear_and_nonlinear_observations_keep_their_real_point_contract(nonlinear, endpoint):
    _case, program, blocks, _states, problem, observed, field, point = _baseline(
        nonlinear=nonlinear, endpoint=endpoint)
    value = observed[field[problem.unknowns[0]]]
    program.store_history("admitted-point", value, depth=1, owner_block=blocks[0])
    assert value.point is point and value.block is None
    assert program._serialize()["version"] == 16


def test_resealed_existing_owner_does_not_remint_the_original_storage_authority():
    from pops.time._program.global_history_storage import storage_contract
    _case, program, blocks, _states, _problem, value, _field, _point, owner, _issued = _storage_only(3)
    node = program.store_history("immutable-owner", value, depth=1, owner_block=owner)
    assert program._serialize()["version"] == 16
    # A real, already declared State from the same Case and clock has the same
    # physical frame. Replace all mutable projections coherently; an independent
    # issued-storage proof must still bind the original owner of this ring.
    other = blocks[1]
    replacement = storage_contract(program, value, other)
    object.__setattr__(node, "block", other)
    object.__setattr__(node, "attrs", dict(node.attrs) | {"global_field_storage": replacement})
    program._history_blocks["immutable-owner"] = other
    with pytest.raises(ValueError):
        program._ir_hash()


def test_removing_storage_contract_cannot_downgrade_an_issued_ring_to_legacy():
    _case, program, *_prefix, value, _field, _point, owner, _issued = _storage_only(3)
    node = program.store_history("no-upcast", value, depth=1, owner_block=owner)
    assert program._serialize()["version"] == 16
    object.__setattr__(node, "attrs", {key: item for key, item in node.attrs.items()
                                     if key != "global_field_storage"})
    with pytest.raises(ValueError):
        program._serialize()


def _cpp_method(text, signature):
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def test_authentic_native_guard_and_collective_helper_with_host_storage_and_transport(tmp_path):
    """Actual C++ guard/helper; storage, reduction and communicator are substitutes.

    This receives branching/validation/ordering, not Kokkos/MPI native execution.
    """
    root = Path(__file__).resolve().parents[2]
    directory = root / "include/pops/runtime/program"
    public = (directory / "amr_program_context_history_checkpoint_public.inc").read_text()
    services = (directory / "amr_program_context_history_checkpoint_services.inc").read_text()
    collective = (directory / "amr_program_context_flux_expression_services.inc").read_text()
    methods = "\n".join((
        _cpp_method(collective, "void require_prepared_lane_("),
        "template <class Build, class Contract>\n" + _cpp_method(collective, "auto prepare_history_mutation_collectively_("),
        _cpp_method(services, "static void require_same_layout_("),
        _cpp_method(services, "static void require_same_field_contract_("),
        _cpp_method(services, "void require_current_boundary_point_exact_("),
        _cpp_method(services, "static std::string history_key_("),
        _cpp_method(services, "void require_history_owner_("),
        _cpp_method(public, "void store_global_field_history("),
    ))
    source = tmp_path / "actual_guard.cpp"
    source.write_text(_HOST_PREFIX + methods + _HOST_SUFFIX)
    executable = tmp_path / "actual_guard"
    subprocess.run(["/usr/bin/clang++", "-std=c++20", "-pthread", "-O0", str(source), "-o", str(executable)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "actual guard: 41 host counterchecks passed"


_HOST_PREFIX = r'''
#include <algorithm>
#include <array>
#include <barrier>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#define POPS_HD
using Real=double;
template<int> using Index=int;
namespace Kokkos { bool isfinite(double x) { return std::isfinite(x); } }
thread_local bool launch_fault=false, fence_fault=false, contract_fault=false;
thread_local int fences=0;
void device_fence() { ++fences; if(fence_fault) throw std::runtime_error("fence fault"); }
template<class F> double for_each_cell_reduce_max(int cells,F fn) {
  if(launch_fault) throw std::runtime_error("launch fault");
  double result=0; for(int i=0;i<cells;++i) result=std::max(result,fn(i)); return result;
}
struct Fraction { long numerator=1,denominator=1; double value()const{return double(numerator)/denominator;}
  bool operator==(const Fraction&)const=default; };
namespace runtime::multiblock { struct BoundaryEvaluationPoint {
 std::string clock="macro"; long tick=4; int level=0,substep=0,stage=7;
 Fraction stage_fraction; double dt=.1,physical_time=.1;
 std::string graph_identity,rate_identity,application_identity;
}; }
struct View { const std::vector<double>* data; double operator()(int cell,int)const{return data->at(cell);} };
struct Fab { std::vector<double> data{2}; View view()const{return {&data};} };
struct Field {
 int layout_id=8,distribution_id=3,rank=0,components=1;
 std::array<int,2> halo{1,1}; std::vector<Fab> fabs{Fab{}};
 int layout()const{return layout_id;} int distribution()const{return distribution_id;}
 int local_rank()const{return rank;} std::size_t local_size()const{return fabs.size();}
 int ncomp()const{return components;} auto ghosts()const{return halo;}
 const Fab& fab(std::size_t i)const{return fabs.at(i);} int box(std::size_t i)const{return fabs.at(i).data.size();}
};
struct Manager {
 std::map<std::string,std::vector<Field>> histories;
 std::map<std::string,int> owner,depth;
 std::map<std::string,std::string> state_identity,space_identity,clock_identity;
};
struct Runtime { Manager hist_; };
struct Facade {
 std::array<Field,2> states; long macro_step()const{return 4;}
 const Field& prepared_amr_block_state(int owner,int)const{return states.at(owner);}
 std::string prepared_amr_block_state_identity_(std::size_t owner)const{return owner==0?"issued-State0":"issued-State1";}
};
struct Transport {
 std::barrier<> barrier{2}; std::array<long,2> flags{};
 std::array<std::exception_ptr,2> errors{}; std::array<std::string,2> contracts;
 std::array<int,2> votes{},agreements{};
};
struct ExecutionLane {
 Transport* transport=nullptr; int rank=0;
 bool active()const{return true;} std::string identity()const{return "prepared-lane";}
 bool congruent_with(const ExecutionLane& rhs)const{return transport==rhs.transport;}
};
long all_reduce_max(long flag,const ExecutionLane& lane) {
 auto& t=*lane.transport; t.flags[lane.rank]=flag; t.barrier.arrive_and_wait();
 long result=std::max(t.flags[0],t.flags[1]); t.barrier.arrive_and_wait(); return result;
}
void collectively_rethrow_exception(std::exception_ptr error,const ExecutionLane& lane,const std::string&) {
 auto& t=*lane.transport; ++t.votes[lane.rank]; t.errors[lane.rank]=error; t.barrier.arrive_and_wait();
 bool failure=bool(t.errors[0])||bool(t.errors[1]); t.barrier.arrive_and_wait();
 if(failure) throw std::runtime_error("host collective refusal");
}
bool all_ranks_agree_exact_ordered_byte_pairs(
 const std::vector<std::pair<std::string_view,std::string_view>>& rows,const ExecutionLane& lane) {
 auto& t=*lane.transport; ++t.agreements[lane.rank]; t.contracts[lane.rank]=std::string(rows.at(0).second);
 t.barrier.arrive_and_wait(); bool same=t.contracts[0]==t.contracts[1]; t.barrier.arrive_and_wait(); return same;
}
struct ExactContractBuilder {
 std::ostringstream out;
 ExactContractBuilder& bytes(const std::string& value) {
  if(contract_fault) throw std::bad_alloc(); out<<value.size()<<":"<<value<<";"; return *this;
 }
 template<class T> ExactContractBuilder& scalar(T value){out<<value<<";";return *this;}
 std::string release()&&{return out.str();}
};
struct Context {
 static constexpr int Dim=2; using field_type=Field;
 mutable Runtime runtime; Facade facade; Facade* facade_=&facade;
 ExecutionLane lane; int active_level_=0,logical_substep_=0;
 std::string primary_clock_="macro"; double current_dt_=.1,current_interval_start_time_=0;
 Fraction stage_time_; mutable int stores=0;
 Context(Transport& t,int rank):lane{&t,rank} {
  for(auto& state:facade.states){state.rank=rank;state.components=5;}
  auto& m=runtime.hist_; std::string key="pops.amr.level-history.v1/0/4:ring";
  Field ring;ring.rank=rank; m.histories[key]={ring};m.owner[key]=0;m.depth[key]=1;
  m.state_identity[key]="issued-descriptor";m.space_identity[key]="scalar-output-field-v1";m.clock_identity[key]="macro";
 }
 const ExecutionLane& prepared_execution_lane()const{return lane;}
 int sys_block(int owner)const{return owner>=0&&owner<2?owner:-1;}
 Runtime& runtime_state()const{return runtime;}
 static void require_rate_identity_(int stage){if(stage<0)throw std::invalid_argument("stage");}
 void store_history_(const std::string&,const field_type&)const{++stores;}
'''

_HOST_SUFFIX = r'''
};
int main() {
 int checks=0;
 // In every negative case only rank zero is altered. Actual preparation must
 // vote before agreement/store; both host ranks refuse with no publication.
 for(int attack=0;attack<40;++attack) {
  Transport transport; std::array<bool,2> refused{};
  std::array<int,2> stores{},fence_counts{};
  std::array<std::thread,2> threads;
  for(int rank=0;rank<2;++rank) threads[rank]=std::thread([&,rank]{
   Context ctx(transport,rank); Field value;value.rank=rank;
   std::string key="pops.amr.level-history.v1/0/4:ring",descriptor="issued-descriptor",witness="issued-State0";
   int owner=0;runtime::multiblock::BoundaryEvaluationPoint point;
   auto& m=ctx.runtime.hist_;
   if(rank==0) switch(attack) {
    case 1:owner=1;break; case 2:owner=-1;break; case 3:owner=9;break;
    case 4:m.owner[key]=1;break;
    case 5:owner=1;m.owner[key]=1;descriptor="resealed";m.state_identity[key]=descriptor;break;
    case 6:witness.clear();break; case 7:witness="issued-State1";break;
    case 8:m.histories.clear();break; case 9:m.histories[key].clear();break;
    case 10:m.depth[key]=0;break; case 11:m.depth[key]=2;break;
    case 12:m.state_identity[key]="other";break;case 13:descriptor.clear();break;
    case 14:m.space_identity[key]="other";break;case 15:m.clock_identity[key]="other";break;
    case 16:value.layout_id=4;break;case 17:value.distribution_id=4;break;case 18:value.rank=1;break;
    case 19:value.fabs.clear();break;case 20:value.components=2;break;
    case 21:m.histories[key].front().components=2;break;case 22:m.histories[key].front().halo[0]=2;break;
    case 23:value.fabs[0].data[0]=NAN;break;case 24:value.fabs[0].data[0]=INFINITY;break;
    case 25:point.clock="child";break;case 26:point.tick++;break;case 27:point.level++;break;
    case 28:point.substep++;break;case 29:point.stage=-1;break;case 30:point.stage_fraction.numerator=2;break;
    case 31:point.dt=std::nextafter(.1,1.);break;case 32:point.physical_time=std::nextafter(.1,1.);break;
    case 33:point.graph_identity="foreign";break;case 34:point.rate_identity="foreign";break;
    case 35:point.application_identity="foreign";break;
    case 36:launch_fault=true;break;case 37:fence_fault=true;break;case 38:contract_fault=true;break;
    case 39:point.stage++;break; // Locally valid but exact operation contracts disagree.
   }
   try{ctx.store_global_field_history("ring",value,owner,descriptor,witness,point);}
   catch(...){refused[rank]=true;}
   stores[rank]=ctx.stores;fence_counts[rank]=fences;
  });
  for(auto& thread:threads)thread.join();
  for(int rank=0;rank<2;++rank) {
   assert(refused[rank]==(attack!=0));assert(stores[rank]==(attack==0));
   assert(transport.votes[rank]==1);
   assert(transport.agreements[rank]==(attack==0||attack==39));
   if(attack==36||attack==37)assert(fence_counts[0]>=1);
  }
  ++checks;
 }
 // A genuinely empty peer for this host storage/transport seam remains legal.
 Transport transport;std::array<int,2> stores{};std::array<std::thread,2> threads;
 for(int rank=0;rank<2;++rank)threads[rank]=std::thread([&,rank]{
  Context ctx(transport,rank);Field value;value.rank=rank;
  if(rank==1){value.fabs.clear();for(auto& field:ctx.facade.states)field.fabs.clear();
    ctx.runtime.hist_.histories.begin()->second.front().fabs.clear();}
  ctx.store_global_field_history("ring",value,0,"issued-descriptor","issued-State0",{});stores[rank]=ctx.stores;
 });
 for(auto& thread:threads)thread.join();assert(stores[0]==1&&stores[1]==1);++checks;
 assert(checks==41);
 std::cout<<"actual guard: "<<checks<<" host counterchecks passed\n";
}
'''
