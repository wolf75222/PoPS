"""Exact-SHA ALE authority/control-flow probes, with host storage/collective seams.

No installation, native evolution, MPI or Kokkos execution is performed. The two
full production methods and preparation guard are extracted without modification;
carrier privacy, attempt leases, points and contract serialization use real headers.
"""
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
SHA = "0b27b20240a3a23061a2197e262b603783a6d258"
ARCHIVE_HASH = "def4e9b962e340ae0aa6e400365bcd7eb657f738ddbddcb3992c39dbae549df8"
OUT = ROOT / "outputs/ale-sol61-0b27b20-source"
archive = subprocess.check_output([
    "git", "archive", SHA, "include", "src/runtime/system/system.cpp"
], cwd=ROOT)
assert sha256(archive).hexdigest() == ARCHIVE_HASH
OUT.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
    bundle.extractall(OUT, filter="data")
fragment_path = OUT / "include/pops/runtime/program/program_context_moving_interval.inc"
source = fragment_path.read_text()
evaluate = source[source.index("  template <class Producer>\n  RuntimeIntervalEvaluation"):
                  source.index("  /// Prepare a coupled SSA")]
commit = source[source.index("  void commit_moving_interval("):
                source.index("  void advance_moving_intervals(")]
guard = source[source.index("        if (evaluation.owner_"):
               source.index("        mf_arith_detail::require_same_layout(input,")]
authority = {
    "commit": SHA, "archive_sha256": ARCHIVE_HASH,
    "fragments": {name: sha256(body.encode()).hexdigest()
                  for name, body in {"evaluate": evaluate, "commit": commit,
                                     "prepare_authority_guard": guard}.items()},
}
(OUT / "host-authority.json").write_text(json.dumps(authority, indent=2) + "\n")

# Override only storage and ledger value types. Real authority/cache/carrier headers
# come from the archived commit, not the checkout or installed package.
seams = OUT / "host-seams"
storage = seams / "pops/mesh/storage/multifab.hpp"
storage.parent.mkdir(parents=True, exist_ok=True)
storage.write_text(r'''
#pragma once
#include <array>
#include <vector>
#include <stdexcept>
#include <pops/core/foundation/types.hpp>
namespace pops {
template<int D> using Index=std::array<int,D>;
struct HostBox {};
struct HostView {
  double* value;
  double& operator()(const Index<1>&, int=0) const {return *value;}
};
struct HostFab {
  double value=1;
  HostView view() const {return {const_cast<double*>(&value)};}
};
template<int D> struct MultiFab {
  std::vector<HostFab> values{HostFab{}};
  std::size_t local_size() const {return values.size();}
  int ncomp() const {return 1;}
  HostBox box(std::size_t) const {return {};}
  const HostFab& fab(std::size_t i) const {return values.at(i);}
};
namespace mf_arith_detail {
template<class A> void require_same_layout(const A& a,const A& b,const char*) {
  if(a.values.size()!=b.values.size()) throw std::invalid_argument("layout");
}
}
}
''')
face = seams / "pops/numerics/spatial/nd/face_field.hpp"
face.parent.mkdir(parents=True, exist_ok=True)
face.write_text("#pragma once\nnamespace pops::nd {template<int D> struct FaceField {}; }\n")
exchange = seams / "pops/runtime/program/accepted_exchange.hpp"
exchange.parent.mkdir(parents=True, exist_ok=True)
exchange.write_text("#pragma once\nnamespace pops::runtime::program {struct ExchangeRecord {int value=1;};}\n")

host = r'''
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <optional>
#include <pops/runtime/program/prepared_resource_cache.hpp>
#include <pops/runtime/program/moving_interval_geometry.hpp>
#include <pops/core/identity/prepared_provider.hpp>
namespace Kokkos {inline void fence() {}}
namespace pops::runtime::program {
using Point=multiblock::BoundaryEvaluationPoint;
std::vector<std::string> events;
bool fail_launch=false, fail_contract=false;
void collectively_rethrow_exception(std::exception_ptr error,int,const char* label) {
  events.push_back(label);
  if(error) std::rethrow_exception(error);
}
template<class T> T all_reduce_max(T value,int) {
  events.push_back("max-vote"); return value;
}
bool all_ranks_agree_exact_ordered_byte_pairs(
    std::initializer_list<std::pair<std::string_view,std::string_view>>,int) {
  events.push_back("exact-contract");return !fail_contract;
}
template<class F> double for_each_cell_reduce_max(HostBox,F&& f) {
  if(fail_launch) throw std::runtime_error("local launch fault");
  return f(Index<1>{0});
}
struct HostSystem {
  int depth=1;
  bool fail_physical=false, fail_stage=false, eb=false;
  std::vector<ExchangeRecord> ledger;
  int step_transaction_depth() const {return depth;}
  const int* validate_program_state_publication_candidate_(int,const MultiFab<1>&,int) {
    events.push_back("physical-recovery");
    if(fail_physical) throw std::invalid_argument("injected physical rejection");
    static const int mask=1;return eb?&mask:nullptr;
  }
  void stage_program_exchanges(const std::vector<ExchangeRecord>& records) {
    events.push_back("ledger-stage");
    if(fail_stage) throw std::runtime_error("ledger admission fault");
    ledger=records;
  }
};
template<int Dim> class ProgramContext {
 public:
  using field_type=MultiFab<Dim>;
  struct Runtime {
    std::map<std::string,MovingIntervalGeometry<Dim>> moving_interval_geometry_;
  };
  mutable PreparedResourceCache cache;
  mutable Runtime runtime;
  mutable field_type field;
  HostSystem owned_system;
  HostSystem* system_=&owned_system;
  ::pops::amr::Rational stage_time_{0,1},logical_phase_begin_{0,1},logical_phase_span_{1,1};
  mutable Point point;
  ProgramContext() {
    cache.begin_attempt();
    point.clock="clock";point.dt=.1;point.physical_time=0;
    auto& geometry=runtime.moving_interval_geometry_["geometry"];
    geometry.runtime_block=0;geometry.physical_frame="frame";
  }
  int prepared_execution_lane() const {return 0;}
  Runtime& runtime_state() const {return runtime;}
  field_type& state(int) const {return field;}
  int sys_block(int p) const {return p;}
  const MovingIntervalGeometry<Dim>& moving_interval_geometry(const std::string& id) const {
    return runtime.moving_interval_geometry_.at(id);
  }
  PreparedResourceAttempt resource_attempt() const {return cache.current_attempt();}
  Point boundary_evaluation_point(int) const {return point;}
'''
host += evaluate + commit
host += r'''
  void probe_prepare_authority(const RuntimeIntervalEvaluation<Dim>& evaluation) const {
    const auto& accepted=moving_interval_geometry(evaluation.identity_);
'''
host += guard
host += r'''
  }
  // Supply a candidate at the tested commit seam; numeric preparation is not faked
  // into a claimed native run. Its previous state/authority come from the real issue.
  PreparedMovingIntervalUpdate<Dim> probe_candidate(const RuntimeIntervalEvaluation<Dim>& evaluation) const {
    probe_prepare_authority(evaluation);
    PreparedMovingIntervalUpdate<Dim> p;
    p.owner_=this;p.attempt_=evaluation.attempt_;p.identity_=evaluation.identity_;
    p.physical_frame_=evaluation.physical_frame_;p.quadrature_=evaluation.quadrature_;
    p.program_block_=evaluation.program_block_;p.point_=evaluation.point_;
    p.geometry_=moving_interval_geometry(p.identity_);++p.geometry_.generation;
    p.geometry_.last_interval="actual-point";
    p.initial_state_=evaluation.initial_state_;p.state_=evaluation.initial_state_;
    p.state_.values[0].value=2;p.records_.push_back({});return p;
  }
  bool published(const PreparedMovingIntervalUpdate<Dim>& p) const {return p.published_;}
};
using Ctx=ProgramContext<1>;
static_assert(!std::is_default_constructible_v<RuntimeIntervalEvaluation<1>>);
static_assert(!std::is_default_constructible_v<PreparedMovingIntervalUpdate<1>>);
static_assert(!std::is_copy_constructible_v<PreparedMovingIntervalUpdate<1>>);
auto issue(Ctx& c) {
  return c.evaluate_moving_interval("geometry",0,"frame","quad",[&](const Point& p) {
    assert(p==c.point);return MovingIntervalInputs<1>{};
  });
}
int checks=0;
template<class F> void refused(const char* label,F&& f) {
  bool rejected=false;try {f();} catch(const std::exception&) {rejected=true;}
  if(!rejected) {std::cerr<<"not rejected: "<<label<<"\n";std::abort();}
  ++checks;
}
void unchanged(const Ctx& c,double value=1) {
  assert(c.field.values[0].value==value);
  assert(c.runtime.moving_interval_geometry_.at("geometry").generation==0);
  assert(c.owned_system.ledger.empty());
}
void probes() {
  {Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);c.commit_moving_interval(p);
   assert(c.published(p)&&c.field.values[0].value==2&&c.owned_system.ledger.size()==1);
   assert(c.runtime.moving_interval_geometry_.at("geometry").generation==1);
   assert(std::find(events.begin(),events.end(),"physical-recovery")<
          std::find(events.begin(),events.end(),"ledger-stage"));
   refused("reuse proposal",[&]{c.commit_moving_interval(p);});
   assert(c.owned_system.ledger.size()==1);++checks;}
  {Ctx c,other;auto e=issue(c);auto p=c.probe_candidate(e);
   refused("foreign owner prepare",[&]{other.probe_prepare_authority(e);});
   refused("foreign owner commit",[&]{other.commit_moving_interval(p);});unchanged(other);}
  {Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);auto old=c.resource_attempt();
   c.cache.begin_attempt();assert(!old.visible());
   refused("old lease prepare",[&]{c.probe_prepare_authority(e);});
   refused("old lease commit",[&]{c.commit_moving_interval(p);});unchanged(c);}
  {Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);c.cache.reject_attempt();
   refused("revoked lease",[&]{c.commit_moving_interval(p);});unchanged(c);}
  for(int mutation=0;mutation!=5;++mutation) {
    Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);
    if(mutation==0)c.point.dt=.15;
    if(mutation==1)++c.point.tick;
    if(mutation==2)c.point.physical_time=.1;
    if(mutation==3)c.point.graph_identity="other graph";
    if(mutation==4)c.point.stage_fraction={1,2};
    refused("stale exact point prepare",[&]{c.probe_prepare_authority(e);});
    refused("stale exact point commit",[&]{c.commit_moving_interval(p);});unchanged(c);
  }
  for(int mutation=0;mutation!=3;++mutation) {
    Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);
    if(mutation==0)c.stage_time_={1,2};
    if(mutation==1)c.logical_phase_begin_={1,2};
    if(mutation==2)c.logical_phase_span_={1,2};
    refused("partial issue",[&]{issue(c);});
    refused("partial prepare",[&]{c.probe_prepare_authority(e);});
    refused("partial commit",[&]{c.commit_moving_interval(p);});unchanged(c);
  }
  {Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);
   ++c.runtime.moving_interval_geometry_.at("geometry").generation;
   refused("stale generation prepare",[&]{c.probe_prepare_authority(e);});
   refused("stale generation commit",[&]{c.commit_moving_interval(p);});
   assert(c.owned_system.ledger.empty()&&c.field.values[0].value==1);}
  {Ctx c;auto e=issue(c);c.runtime.moving_interval_geometry_.at("geometry").physical_frame="other";
   refused("changed frame prepare",[&]{c.probe_prepare_authority(e);});unchanged(c);}
  {Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);c.field.values[0].value=100;
   refused("changed accepted input",[&]{c.commit_moving_interval(p);});unchanged(c,100);}
  {Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);c.owned_system.fail_physical=true;
   refused("physical recovery",[&]{c.commit_moving_interval(p);});unchanged(c);}
  {Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);c.owned_system.eb=true;
   refused("static EB",[&]{c.commit_moving_interval(p);});unchanged(c);}
  {Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);fail_launch=true;events.clear();
   refused("local launch collective",[&]{c.commit_moving_interval(p);});fail_launch=false;
   assert(events.back()=="moving SSA input validation failed collectively");
   assert(std::find(events.begin(),events.end(),"max-vote")==events.end());unchanged(c);}
  {Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);fail_contract=true;
   refused("rank contract mismatch",[&]{c.commit_moving_interval(p);});fail_contract=false;unchanged(c);}
  {Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);c.owned_system.fail_stage=true;
   refused("ledger admission",[&]{c.commit_moving_interval(p);});unchanged(c);assert(!c.published(p));}
  {Ctx c;refused("callback changes point",[&]{
     c.evaluate_moving_interval("geometry",0,"frame","quad",[&](const Point&) {
       c.point.dt=.15;return MovingIntervalInputs<1>{};});});unchanged(c);}
  {Ctx c;refused("callback failure vote",[&]{
     c.evaluate_moving_interval("geometry",0,"frame","quad",[](const Point&)->MovingIntervalInputs<1> {
       throw std::runtime_error("producer fault");});});
   assert(events.back()=="moving interval evaluator failed collectively");unchanged(c);}
  {Ctx c;refused("callback changes generation",[&]{
     c.evaluate_moving_interval("geometry",0,"frame","quad",[&](const Point&)->MovingIntervalInputs<1> {
       ++c.runtime.moving_interval_geometry_.at("geometry").generation;return {};});});
   assert(c.owned_system.ledger.empty()&&c.field.values[0].value==1);}
  {Ctx c;auto e=c.evaluate_moving_interval("geometry",0,"frame","quad",[&](const Point&)->MovingIntervalInputs<1> {
       c.cache.begin_attempt();return {};});
   refused("callback superseded attempt",[&]{c.probe_prepare_authority(e);});unchanged(c);}
  {Ctx c;refused("wrong frame issue",[&]{
     c.evaluate_moving_interval("geometry",0,"wrong frame","quad",[](const Point&)->MovingIntervalInputs<1> {return {};});});unchanged(c);}
  {Ctx c;refused("empty quadrature issue",[&]{
     c.evaluate_moving_interval("geometry",0,"frame","",[](const Point&)->MovingIntervalInputs<1> {return {};});});unchanged(c);}
  {Ctx c;c.owned_system.depth=0;
   refused("no transaction",[&]{issue(c);});unchanged(c);}
  std::cout<<checks<<" exact-method host checks passed\n";
  // Confirm the independently found gap on this frozen SHA. This intentionally
  // records acceptance where fail-closed behavior is required, not a passing guard.
  for(int mutation=0;mutation!=2;++mutation) {
    Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);
    auto& accepted=c.runtime.moving_interval_geometry_.at("geometry");
    if(mutation==0)accepted.physical_frame="changed after preparation";
    if(mutation==1)accepted.measures.values[0].value=7;
    c.commit_moving_interval(p);
    assert(c.published(p)&&c.owned_system.ledger.size()==1);
    assert(accepted.physical_frame=="frame"&&accepted.measures.values[0].value==1);
  }
  std::cout<<"CONFIRMED GAP: changed accepted frame/measure at constant generation is overwritten (2 probes)\n";
}
}
int main() {pops::runtime::program::probes();}
'''
translation_unit = OUT / "interval_authority_host.cpp"
translation_unit.write_text(host)
executable = OUT / "interval_authority_host"
compiler = shutil.which("clang++") or shutil.which("c++")
assert compiler, "host C++20 compiler required"
result = subprocess.run([
    compiler, "-std=c++20", "-O0", "-I" + str(seams), "-I" + str(OUT / "include"),
    str(translation_unit), "-o", str(executable),
], capture_output=True, text=True)
if result.returncode:
    raise RuntimeError(result.stderr[-10000:])
subprocess.run([str(executable)], check=True)
print("Exact archive:", SHA, ARCHIVE_HASH)
print("Host seams: scalar storage, launcher, serial collective voting, physics and ledger.")
