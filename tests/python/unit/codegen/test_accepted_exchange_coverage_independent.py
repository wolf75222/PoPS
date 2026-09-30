"""Run production exchange producers against explicit host masks, without a PDE/JIT.

The geometry/field adapter is deliberately small; ExchangeRecord is the production
header. Native AMR preparation and MPI still require the installed reception.
"""
import shutil
import subprocess
from pathlib import Path

import pops
import pytest

from pops.codegen.program_emit_transport_exchanges import emit_transport_exchanges


def _sdk_root():
    package = Path(pops.__file__).resolve().parent
    if (package / "include/pops/runtime/program/accepted_exchange.hpp").is_file():
        return package
    source = Path(__file__).resolve().parents[4]
    assert Path(pops.__file__).resolve() == source / "python/pops/__init__.py"
    return source


HOST = r'''#include <pops/runtime/program/accepted_exchange.hpp>
#include <array>
#include <iostream>
#include <type_traits>
namespace pops {
using Real=double;
constexpr int kNativeDimension=2;
using Index=std::array<int,2>;
struct Box {
  Index lo{0,0}, hi{1,3};
  Index extent() const { return {hi[0]-lo[0]+1,hi[1]-lo[1]+1}; }
  std::int64_t numPts() const { auto e=extent(); return e[0]*e[1]; }
  bool operator==(const Box&) const = default;
};
template<class T,int D> struct FieldView {
  const std::vector<double>* data=nullptr;
  Box box;
  double operator()(Index i,int) const {
    return data->at((i[1]-box.lo[1])*box.extent()[0]+i[0]-box.lo[0]);
  }
};
struct Fab {
  Box bounds; std::vector<double> data;
  FieldView<const Real,2> view() const { return {&data,bounds}; }
};
template<int D> struct MultiFab {
  std::vector<Fab> fields;
  std::size_t local_size() const { return fields.size(); }
  const Box& box(std::size_t i) const { return fields.at(i).bounds; }
  const Fab& fab(std::size_t i) const { return fields.at(i); }
};
struct Lane { int size() const { return 1; } };
inline long all_reduce_max(long value,const Lane&) { return value; }
inline void sync_host() {}
}
using Record=pops::runtime::program::ExchangeRecord;
struct FaceAxis {
  double measure=1;
  double operator()(pops::Index,int component) const { return measure*1.2*(component+1); }
};
struct Faces {
  pops::Box bounds; double hy=.25;
  const pops::Box& cell_box() const { return bounds; }
  int ncomp() const { return 2; }
  struct View { std::array<FaceAxis,2> axes; };
  View view() const { return View{std::array<FaceAxis,2>{FaceAxis{hy},FaceAxis{.5}}}; }
};
struct Context {
  pops::MultiFab<2> field, coverage, embedded;
  std::vector<Faces> faces;
  std::vector<Record> records;
  bool use_coverage=true, use_embedded=false;
  double hy=.25;
  Context(int nx,int ny,bool empty=false) : hy(1./ny) {
    if(empty) return;
    pops::Box box{{0,0},{nx-1,ny-1}};
    pops::Fab fab{box,std::vector<double>(nx*ny,1)};
    field.fields={fab}; coverage.fields={fab}; embedded.fields={fab};
    faces={{box,hy}};
  }
  const pops::MultiFab<2>& state(int) const { return field; }
  const pops::MultiFab<2>* pointwise_active_mask(int,const pops::MultiFab<2>&) const {
    return use_embedded?&embedded:nullptr;
  }
  const pops::MultiFab<2>* pointwise_exchange_coverage_mask(int,const pops::MultiFab<2>&) const {
    return use_coverage?&coverage:nullptr;
  }
  pops::Lane prepared_execution_lane() const { return {}; }
  struct Geometry { double hy; double spacing(int axis) const { return axis?hy:.5; } };
  Geometry geometry() const { return {hy}; }
  bool is_external_trace_face(int axis,int side,pops::Index cell) const {
    return axis==0 && side==1 && cell[0]==field.fields[0].bounds.hi[0];
  }
  template<class Body> void stage_exchange_batch(Body body) {
    auto candidate=records;
    body([&](Record record){
      record.source_evaluation_identity=record.evaluation_context;
      record.validate(); candidate.push_back(std::move(record));
    });
    records.swap(candidate);
  }
  void cover_right(int first,int end) {
    auto& fab=coverage.fields[0]; const auto nx=fab.bounds.extent()[0];
    for(int y=first;y<end;++y) fab.data[y*nx+nx-1]=0;
  }
};
void emit(Context& ctx,double dt) {
  const auto& faces=ctx.faces;
@@EMITTER@@
}
std::pair<int,double> selected(const Context& ctx,int component=0) {
  int count=0; double amount=0;
  for(const auto& r:ctx.records) if(r.exterior_trace && r.trace_axis==0 &&
      r.trace_side==1 && r.trace_component==component) {
    ++count; amount+=r.integrated_amount();
  }
  return {count,amount};
}
void close(double actual,double expected) {
  if(std::abs(actual-expected)>2e-15) throw std::runtime_error("wrong accepted trace amount");
}
void count(int actual,int expected) {
  if(actual!=expected) throw std::runtime_error("covered cell published a trace record");
}
int main() {
  Context coarse(2,4),fine(4,8);
  coarse.cover_right(0,4);
  emit(coarse,.01); emit(fine,.005); emit(fine,.005);
  count(selected(coarse).first,0); count(selected(fine).first,16);
  close(.7-selected(coarse).second-selected(fine).second,.712);
  close(selected(fine,1).second,-.024);
  // Interface intersects the physical trace: only half belongs to each level.
  Context mixed_coarse(2,4),mixed_fine(4,8);
  mixed_coarse.cover_right(2,4); mixed_fine.cover_right(0,4);
  emit(mixed_coarse,.01); emit(mixed_fine,.005); emit(mixed_fine,.005);
  count(selected(mixed_coarse).first,2); count(selected(mixed_fine).first,8);
  close(selected(mixed_coarse).second,-.006);
  close(selected(mixed_fine).second,-.006);
  // EB and coverage are independent authorities and must be intersected.
  Context both(2,4); both.cover_right(2,4); both.use_embedded=true;
  both.embedded.fields[0].data[1]=0; emit(both,.01);
  count(selected(both).first,1); close(selected(both).second,-.003);
  // No local patches is legal; the global presence vote is another seam.
  Context empty(2,4,true); emit(empty,.01); count(empty.records.size(),0);
  // Uniform retains the prior no-coverage behavior.
  Context uniform(2,4); uniform.use_coverage=false; emit(uniform,.01);
  count(selected(uniform).first,4); close(selected(uniform).second,-.012);
  // A foreign mask cannot publish a partial batch.
  Context foreign(2,4); foreign.coverage.fields[0].bounds.hi[0]=0;
  std::string diagnostic;
  try { emit(foreign,.01); } catch(const std::invalid_argument& e) { diagnostic=e.what(); }
  if(diagnostic.find("mask differs from local patches")==std::string::npos)
    throw std::runtime_error("foreign coverage was not rejected precisely");
  count(foreign.records.size(),0);
  std::cout << "transport coverage host PASS\n";
}
'''


def test_actual_transport_emitter_intersects_coverage_and_embedded_boundary(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++20 compiler required")
    root = _sdk_root()
    generated = "\n".join(emit_transport_exchanges(
        "faces", "operation", "occurrence", "stage0/evaluation0", "dt"))
    source = tmp_path / "coverage.cpp"
    source.write_text(HOST.replace("@@EMITTER@@", generated))
    executable = tmp_path / "coverage"
    compiled = subprocess.run([compiler, "-std=c++20", "-O2", "-I", str(root / "include"),
                    str(source), "-o", str(executable)], check=False,
                   capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(executable)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "transport coverage host PASS\n"


def test_actual_diffusion_producer_intersects_coverage_and_embedded_boundary(tmp_path):
    """Execute the actual prepared producer; the scaffold supplies already evaluated faces.

    Diffusion retains incidence identities, rather than transport's exterior trace
    metadata. Select x+ by those exact identities and exercise its boundary-only path.
    This does not exercise native field storage, face evaluation, AMR preparation or MPI.
    """
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++20 compiler required")
    root = _sdk_root()
    header = (root / "include/pops/numerics/diffusion/prepared_diffusion.hpp").read_text()
    start = header.index("  template <class Context>\n  void stage_accepted_exchanges(")
    end = header.index("\n  const auto& faces()", start)
    producer = header[start:end].replace("Index<Dim>", "pops::Index")
    scaffold = HOST.split("void emit(Context& ctx,double dt)")[0]
    # Transport faces store spatial integrals; prepared diffusion faces store flux density.
    scaffold = scaffold.replace("FaceAxis{hy},FaceAxis{.5}", "FaceAxis{1},FaceAxis{1}")
    source = tmp_path / "diffusion_coverage.cpp"
    source.write_text(scaffold + r'''
using pops::Real;
using pops::FieldView;
using pops::sync_host;
using pops::all_reduce_max;
enum class DiffusiveBoundaryKind { periodic, prescribed };
struct Boundary { DiffusiveBoundaryKind kind=DiffusiveBoundaryKind::periodic; };
struct DiffusionGeometry {
  pops::Box bounds; double hy;
  double spacing(int axis) const { return axis?hy:.5; }
  const pops::Box& domain() const { return bounds; }
};
struct Producer {
  static constexpr int Dim=2, Components=2;
  using Field=pops::MultiFab<2>;
  const Field& variable_;
  const std::vector<Faces>& faces_;
  DiffusionGeometry geometry_;
  std::array<Boundary,4> physical_{};
  const pops::Lane* lane_;
  Real explicit_frequency() const { return 0; }
''' + producer + r'''
};
void emit(Context& ctx,double dt) {
  pops::Lane lane;
  const auto domain=ctx.field.local_size()?ctx.field.box(0):pops::Box{};
  Producer producer{ctx.field,ctx.faces,{domain,ctx.hy},{},&lane};
  producer.physical_[1].kind=DiffusiveBoundaryKind::prescribed;
  producer.stage_accepted_exchanges(ctx,0,"operation","occurrence","stage0/evaluation0",dt,true);
}
std::pair<int,double> selected(const Context& ctx,int component=0) {
  int count=0; double amount=0;
  const auto suffix="/axis:0/side:1/component:"+std::to_string(component);
  for(const auto& r:ctx.records) if(r.quadrature_identity.ends_with(suffix)) {
    ++count; amount+=r.integrated_amount();
  }
  return {count,amount};
}
void close(double actual,double expected) {
  if(std::abs(actual-expected)>2e-15) throw std::runtime_error("wrong accepted diffusive amount");
}
void count(int actual,int expected) {
  if(actual!=expected) throw std::runtime_error("covered cell published a diffusive record");
}
int main() {
  Context coarse(2,4),fine(4,8);
  coarse.cover_right(0,4);
  emit(coarse,.01); emit(fine,.005); emit(fine,.005);
  count(selected(coarse).first,0); count(selected(fine).first,16);
  close(selected(coarse).second+selected(fine).second,.012);
  close(selected(fine,1).second,.024);
  Context mixed_coarse(2,4),mixed_fine(4,8);
  mixed_coarse.cover_right(2,4); mixed_fine.cover_right(0,4);
  emit(mixed_coarse,.01); emit(mixed_fine,.005); emit(mixed_fine,.005);
  count(selected(mixed_coarse).first,2); count(selected(mixed_fine).first,8);
  close(selected(mixed_coarse).second,.006); close(selected(mixed_fine).second,.006);
  Context both(2,4); both.cover_right(2,4); both.use_embedded=true;
  both.embedded.fields[0].data[1]=0; emit(both,.01);
  count(selected(both).first,1); close(selected(both).second,.003);
  Context empty(2,4,true); emit(empty,.01); count(empty.records.size(),0);
  Context uniform(2,4); uniform.use_coverage=false; emit(uniform,.01);
  count(selected(uniform).first,4); close(selected(uniform).second,.012);
  Context foreign(2,4); foreign.coverage.fields[0].bounds.hi[0]=0;
  std::string diagnostic;
  try { emit(foreign,.01); } catch(const std::invalid_argument& e) { diagnostic=e.what(); }
  if(diagnostic.find("mask differs from local patches")==std::string::npos)
    throw std::runtime_error("foreign diffusive coverage was not rejected precisely");
  count(foreign.records.size(),0);
  std::cout << "diffusion coverage host PASS\n";
}
''')
    executable = tmp_path / "diffusion_coverage"
    compiled = subprocess.run([compiler, "-std=c++20", "-O2", "-I", str(root / "include"),
                               str(source), "-o", str(executable)], check=False,
                              capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(executable)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "diffusion coverage host PASS\n"


def test_production_coverage_lookup_does_not_reenter_collective_preparation(tmp_path):
    """Compile the actual accessor; any second refresh is an explicit test failure.

    The preceding active-mask lookup can fail locally after its preparation. Other
    ranks must not enter another collective before the producer's error vote.
    """
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++20 compiler required")
    root = _sdk_root()
    header = (root / "include/pops/runtime/program/"
              "amr_program_context_history_checkpoint_public.inc").read_text()
    start = header.index("const field_type* pointwise_exchange_coverage_mask(")
    end = header.index("\n}", start) + 2
    accessor = header[start:end]
    source = tmp_path / "prepared_coverage.cpp"
    source.write_text(HOST.split("void emit(Context& ctx,double dt)")[0] + r'''
struct Facade {
  Context* context;
  const pops::MultiFab<2>& prepared_amr_block_state(int,int) const { return context->field; }
  const pops::MultiFab<2>& prepared_amr_block_level_coverage_mask(int,int) const {
    return context->coverage;
  }
};
struct Probe {
  using field_type=pops::MultiFab<2>;
  Facade* facade_; int active_level_=0;
  void refresh_resources_() const {
    throw std::runtime_error("coverage lookup reentered collective preparation");
  }
  int sys_block(int id) const { return id; }
  static void require_same_layout_(const field_type& a,const field_type& b,const char*) {
    if(a.local_size()!=b.local_size()) throw std::invalid_argument("wrong coverage layout");
    for(std::size_t i=0;i<a.local_size();++i)
      if(a.box(i)!=b.box(i)) throw std::invalid_argument("wrong coverage layout");
  }
''' + accessor + r'''
};
int main() {
  Context context(2,4); Facade facade{&context}; Probe probe{&facade};
  if(probe.pointwise_exchange_coverage_mask(0,context.field)!=&context.coverage) return 2;
  context.coverage.fields[0].bounds.hi[0]=0;
  try { probe.pointwise_exchange_coverage_mask(0,context.field); return 3; }
  catch(const std::invalid_argument&) {}
}
''')
    executable = tmp_path / "prepared_coverage"
    compiled = subprocess.run([compiler, "-std=c++20", "-O2", "-I", str(root / "include"),
                               str(source), "-o", str(executable)], check=False,
                              capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(executable)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
