"""Re-receive the corrected ALE authority against immutable prior host scaffolds.

Production methods are extracted unchanged at 745993c. Fixture substitutions
provide the newly required owned receipt and explicit scalar geometry storage.
No native/MPI/Kokkos execution, environment mutation or production edit.
"""
from hashlib import sha256
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SHA = "745993c7a1564f3622639df216569f354bb62c6b"
HASH = "f4a16486a666f437a49c6082d48ef3d1eaf48038ffc847be2a8bc39b3d7ac81a"
BASE = "1981919a0528e5daaa484144277c4023a43c55ec"
fixture = subprocess.check_output([
    "git", "show", BASE + ":tests/review/sol61_ale_0b27b20_authority.py"
], cwd=ROOT, text=True)
base_hash = sha256(fixture.encode()).hexdigest()


def replace(before, after):
    global fixture
    assert fixture.count(before) == 1, before[:100]
    fixture = fixture.replace(before, after)


replace('SHA = "0b27b20240a3a23061a2197e262b603783a6d258"', 'SHA = "' + SHA + '"')
replace('ARCHIVE_HASH = "def4e9b962e340ae0aa6e400365bcd7eb657f738ddbddcb3992c39dbae549df8"',
        'ARCHIVE_HASH = "' + HASH + '"')
replace('OUT = ROOT / "outputs/ale-sol61-0b27b20-source"',
        'OUT = ROOT / "outputs/ale-sol61-745993c-source"')
replace('compiler, "-std=c++20", "-O0",', 'compiler, "-std=c++20", "-O0", "-DPOPS_NATIVE_DIM=1",')
replace('struct HostBox {};',
        'struct HostBox {int tag=0; bool faces=false; friend bool operator==(const HostBox&,const HostBox&)=default;};')
replace('  double* value;\n  double& operator()(const Index<1>&, int=0) const {return *value;}',
        '  double* value; double* indexed=nullptr;\n'
        '  double& operator()(const Index<1>& cell, int=0) const {return indexed?indexed[cell[0]+1]:*value;}')
replace('  double value=1;\n  HostView view() const {return {const_cast<double*>(&value)};}',
        '  double value=1; bool indexed=false; std::array<double,4> data{0,0,1,1};\n'
        '  HostView view() const {return {const_cast<double*>(&value), indexed?const_cast<double*>(data.data()):nullptr};}')
replace('  int ncomp() const {return 1;}',
        '  int components=1; int ncomp() const {return components;}\n  std::array<int,D> ghosts() const {return {1};}')
replace('void require_same_layout(const A& a,const A& b,const char*) {',
        'void require_same_layout(const A& a,const A& b,const char*,bool require_components=true) {')
replace('if(a.values.size()!=b.values.size())',
        'if(a.values.size()!=b.values.size() || (require_components && a.ncomp()!=b.ncomp()))')
replace('namespace pops::nd {template<int D> struct FaceField {}; }',
        'namespace pops::nd {template<int D> struct FaceField { pops::HostFab storage; '
        'pops::HostBox shape; int components=1; '
        'FaceField(){storage.indexed=true;} '
        'FaceField(pops::HostBox box,int width):shape(box),components(width){storage.indexed=true;} '
        'template<int> const pops::HostFab& field() const {return storage;} '
        'pops::HostBox cell_box() const {return shape;} '
        'int ncomp() const {return components;} }; '
        'inline pops::HostBox face_box(pops::HostBox box,int){box.faces=true;return box;} }')
replace('geometry.runtime_block=0;geometry.physical_frame="frame";',
        'geometry.runtime_block=0;geometry.physical_frame="frame";'
        'geometry.coordinates.emplace_back();geometry.swept_volumes.emplace_back();owned_system.live=&field;')
replace('  int depth=1;\n', '  int depth=1;\n  MultiFab<1>* live=nullptr;\n'
        '  MultiFab<1>& block_state(int) {return *live;}\n')
replace('p.geometry_=moving_interval_geometry(p.identity_);++p.geometry_.generation;',
        'p.geometry_=moving_interval_geometry(p.identity_);'
        'p.previous_interval_=p.geometry_.last_interval;'
        'p.geometry_.last_receipt.emplace();'
        'p.geometry_.last_receipt->previous_measures=p.geometry_.measures;'
        'p.geometry_.last_receipt->previous_coordinates=p.geometry_.coordinates;'
        '++p.geometry_.generation;')

context_source = subprocess.check_output([
    "git", "show", SHA + ":include/pops/runtime/program/program_context.hpp"
], cwd=ROOT, text=True)
start = context_source.index("  void commit_many(")
begin = context_source.index("    std::vector<field_type*> targets;", start)
end = context_source.index("    const long commit_count", begin)
classification = context_source[begin:end]
# Compile the unchanged classification and its collective error boundary. The
# full ordinary-copy path is outside this probe; it cannot be reached on refusal.
extra_method = r'''
  void require_same_field_contract_(const field_type& a,const field_type& b,const char* label) const {
    mf_arith_detail::require_same_layout(a,b,label);
  }
  void probe_commit_many(std::initializer_list<std::pair<field_type*,const field_type*>> commits) const {
    struct Lane {int size() const{return 1;} operator int() const{return 0;}} lane;
''' + classification + r'''
  }
'''
projection = subprocess.check_output([
    "git", "show", SHA + ":include/pops/runtime/program/program_context_moving_projection.inc"
], cwd=ROOT, text=True)
extra_method += r'''
  struct ReferenceGeometry {
    double face_coordinate(int,int i) const {return i;}
    double cell_coordinate(int,int i) const {return i+.5;}
  };
  ReferenceGeometry geometry() const {return {};}
  field_type scalar_field_like_(const field_type& input,int,int) const {return input;}
  void fill_boundary(field_type& sampled) const {
    events.push_back("sample-boundary");sampled.values[0].value=3;
  }
  void probe_projection_values(const RuntimeIntervalEvaluation<Dim>& e) const {
    const auto& v=e.inputs_;const double dt=e.point_.dt;
    assert(v.coordinates[0].storage.data[1]==0&&v.coordinates[0].storage.data[2]==1);
    assert(v.swept_volumes[0].storage.data[1]==0&&v.swept_volumes[0].storage.data[2]==0);
    assert(v.physical_flux[0].storage.data[1]==3*dt&&v.physical_flux[0].storage.data[2]==3*dt);
    assert(v.face_density[0].storage.data[1]==3&&v.face_density[0].storage.data[2]==3);
    assert(v.source.values[0].value==3*1.5*dt&&e.initial_state_.values[0].value==1);
  }
''' + projection
replace('namespace Kokkos {inline void fence() {}}',
        'namespace Kokkos {inline void fence() {} inline double abs(double x){return std::abs(x);} '
        'inline bool isfinite(double x){return std::isfinite(x);}}')
replace('#include <algorithm>\n#include <cassert>\n#include <iostream>',
        '#include <algorithm>\n#include <cassert>\n#include <cmath>\n#include <iostream>')
replace('  return f(Index<1>{0});',
        '  return std::max(f(Index<1>{0}), b.faces?f(Index<1>{1}):0.);')
replace('template<class F> double for_each_cell_reduce_max(HostBox,F&& f)',
        'template<class F> double for_each_cell_reduce_max(HostBox b,F&& f)')
replace('struct HostSystem {',
        'template<class F> void for_each_cell(HostBox b,F&& f){f(Index<1>{0});if(b.faces)f(Index<1>{1});}\n'
        'enum class SolveStatus {kInvalidEvaluation};\n'
        'struct StepAttemptRejected:std::runtime_error {StepAttemptRejected(SolveStatus,const char*,const char* message):std::runtime_error(message){}};\n'
        'struct HostSystem {')
replace('  bool published(const PreparedMovingIntervalUpdate<Dim>& p) const {return p.published_;}',
        extra_method + '\n  bool published(const PreparedMovingIntervalUpdate<Dim>& p) const {return p.published_;}')
begin = fixture.index('  // Confirm the independently found gap')
end = fixture.index('\n}\n}\nint main()', begin)
fixture = fixture[:begin] + r'''
  // All four original mutation families now refuse, plus shape/ownership faults.
  for(int mutation=0;mutation!=8;++mutation) {
    Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);
    auto& accepted=c.runtime.moving_interval_geometry_.at("geometry");
    if(mutation==0)accepted.physical_frame="changed after preparation";
    if(mutation==1)accepted.measures.values[0].value=7;
    if(mutation==2)accepted.coordinates[0].storage.data[1]=9;
    if(mutation==3)accepted.last_interval="other interval";
    if(mutation==4)accepted.coordinates.clear();
    if(mutation==5)accepted.measures.values.emplace_back();
    if(mutation==6)accepted.coordinates[0].shape.tag=1;
    if(mutation==7)accepted.coordinates[0].components=2;
    refused("changed accepted geometry at constant generation",[&]{c.commit_moving_interval(p);});
    unchanged(c);assert(!c.published(p));
  }
  {Ctx c;MultiFab<1> trial;trial.values[0].value=2;
   refused("plain density publication",[&]{c.probe_commit_many({{&c.field,&trial}});});unchanged(c);
   refused("identity density publication",[&]{c.probe_commit_many({{&c.field,&c.field}});});unchanged(c);
   c.probe_commit_many({{&trial,&c.field}});++checks;}
  for(double dt:{.1,.3}) {
    Ctx c;c.point.physical_time=.2;c.point.dt=dt;
    auto coordinate=[](double ref,double){return ref;};
    auto face=[](const StateVec<1>& lower,const StateVec<1>& upper,
                  const MovingFaceGeometry&,const MovingIntervalTime& time) {
      MovingFaceEvaluation<1> out;out.physical_amount[0]=lower[0]*time.duration;
      out.density[0]=(lower[0]+upper[0])/2;return out;
    };
    auto cell=[](const StateVec<1>& input,const MovingCellGeometry& g,const MovingIntervalTime& time) {
      StateVec<1> out;out[0]=input[0]*(g.center+1)*time.duration;return out;
    };
    auto e=c.project_moving_interval<1>("geometry",0,"frame","quad",coordinate,face,cell,1e-13);
    assert(e.interval_begin()==.2&&e.interval_end()==.2+dt);c.probe_projection_values(e);unchanged(c);++checks;
    refused("discontinuous coordinate law",[&]{c.project_moving_interval<1>("geometry",0,"frame","quad",
        [](double ref,double){return ref+.1;},face,cell,1e-13);});unchanged(c);
    refused("nonfinite coordinate law",[&]{c.project_moving_interval<1>("geometry",0,"frame","quad",
        [](double,double){return std::numeric_limits<double>::quiet_NaN();},face,cell,1e-13);});unchanged(c);
    c.stage_time_={1,2};refused("partial projection",[&]{c.project_moving_interval<1>("geometry",0,"frame","quad",coordinate,face,cell,1e-13);});unchanged(c);
  }
  std::cout<<checks<<" total corrected authority/classification host checks passed\n";
  {Ctx c;auto e=issue(c);auto p=c.probe_candidate(e);
   auto& accepted=c.runtime.moving_interval_geometry_.at("geometry");
   accepted.measures.components=2;
   c.commit_moving_interval(p);
   assert(c.published(p)&&accepted.measures.ncomp()==1&&c.owned_system.ledger.size()==1);
   std::cout<<"CONFIRMED RESIDUAL GAP: changed accepted measure component width is overwritten\n";}
''' + fixture[end:]

print("Base scaffold:", BASE, sha256(fixture.encode()).hexdigest(), flush=True)
exec(compile(fixture, str(Path(__file__)), "exec"), {"__file__": __file__, "__name__": "__main__"})
receipt = {"commit": SHA, "archive_sha256": HASH, "base_scaffold_commit": BASE,
           "base_scaffold_sha256": base_hash, "adapted_scaffold_sha256": sha256(fixture.encode()).hexdigest(),
           "projection_sha256": sha256(projection.encode()).hexdigest(),
           "commit_many_classification_sha256": sha256(classification.encode()).hexdigest(),
           "received_checks": 61,
           "confirmed_gap": "measure component width changed at constant generation is overwritten"}
(ROOT / "outputs/ale-sol61-745993c-source/review-receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
