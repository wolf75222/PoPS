"""Run production exchange producers against explicit host masks, without a PDE/JIT.

The local topology/rank adapter is deliberately small. Fab, FaceField, host
publication and ExchangeRecord are production headers with real Kokkos storage.
Native AMR preparation and MPI still require the installed reception.
"""
import subprocess
import sys
from pathlib import Path

import pops

from pops.codegen.program_emit_transport_exchanges import emit_transport_exchanges
from tests.python.support.native_syntax_toolchain import configured_kokkos_toolchain


def _sdk_root():
    package = Path(pops.__file__).resolve().parent
    if (package / "include/pops/runtime/program/accepted_exchange.hpp").is_file():
        return package
    source = Path(__file__).resolve().parents[4]
    assert Path(pops.__file__).resolve() == source / "python/pops/__init__.py"
    return source


def _host_toolchain():
    from pops._native_selector import select_native_dimension
    select_native_dimension(2)
    compiler, prefix, flags, link_flags = configured_kokkos_toolchain()
    # A standalone executable does not inherit the already-loaded extension's
    # Kokkos symbols. Link the same authenticated installation explicitly.
    link_flags += ["-L" + str(prefix / "lib"), "-lkokkoscore",
                   "-Wl,-rpath," + str(prefix / "lib")]
    if sys.platform == "darwin":
        link_flags.append("-lomp")
    return compiler, flags, link_flags


HOST = r'''#include <pops/runtime/program/accepted_exchange.hpp>
#include <array>
#include <iostream>
#include <optional>
#include <type_traits>
#include <pops/runtime/program/host_face_publication.hpp>
namespace pops {
constexpr int kNativeDimension=2;
template<int D> struct Geometry {
  double hy;
  double spacing(int axis) const { return axis?hy:.5; }
};
// Only the local topology and rank selection are test adapters. Storage and
// host publication below are the actual PoPS/Kokkos implementation.
template<int D> struct MultiFab {
  std::vector<Fab<D>> fields;
  bool replicated=false;
  struct Distribution {
    bool value;
    bool replicated() const { return value; }
  };
  Distribution distribution() const { return {replicated}; }
  std::size_t local_size() const { return fields.size(); }
  const Box<D>& box(std::size_t i) const { return fields.at(i).box(); }
  const Fab<D>& fab(std::size_t i) const { return fields.at(i); }
};
struct Lane {
  int ordinal=0;
  int ranks=1;
  int size() const { return ranks; }
  int rank() const { return ordinal; }
};
inline long all_reduce_max(long value,const Lane&) { return value; }
inline void sync_host() { device_fence(); }
}
using Record=pops::runtime::program::ExchangeRecord;
using Faces=pops::nd::FaceField<2>;
void set_host_value(pops::Fab<2>& field,std::size_t offset,double value) {
  auto host=field.create_host_mirror(); field.copy_to_host(host);
  host(offset)=value; field.copy_from_host(host);
}
void fill_face(pops::Fab<2>& field,double measure) {
  auto host=field.create_host_mirror();
  const auto count=static_cast<std::size_t>(field.box().numPts());
  for(int component=0;component<field.ncomp();++component)
    for(std::size_t cell=0;cell<count;++cell)
      host(cell+component*count)=measure*1.2*(component+1);
  field.copy_from_host(host);
}
struct Context {
  pops::MultiFab<2> field, coverage, embedded;
  std::vector<Faces> faces;
  std::vector<Record> records;
  bool use_coverage=true, use_embedded=false;
  int rank=0, ranks=1;
  double hy=.25;
  Context(int nx,int ny,bool empty=false) : hy(1./ny) {
    if(empty) return;
    pops::Box<2> box{pops::Index<2>{0,0},pops::Index<2>{nx-1,ny-1}};
    pops::Fab<2> fab(box,1); fab.set_val(1);
    field.fields={fab}; coverage.fields={fab}; embedded.fields={fab};
    faces.emplace_back(box,2);
    fill_face(faces[0].template field<0>(),hy);
    fill_face(faces[0].template field<1>(),.5);
  }
  const pops::MultiFab<2>& state(int) const { return field; }
  const pops::MultiFab<2>* pointwise_active_mask(int,const pops::MultiFab<2>&) const {
    return use_embedded?&embedded:nullptr;
  }
  const pops::MultiFab<2>* pointwise_exchange_coverage_mask(int,const pops::MultiFab<2>&) const {
    return use_coverage?&coverage:nullptr;
  }
  pops::Lane prepared_execution_lane() const { return {rank,ranks}; }
  using Geometry=pops::Geometry<2>;
  Geometry geometry() const { return {hy}; }
  bool is_external_trace_face(int axis,int side,pops::Index<2> cell) const {
    return axis==0 && side==1 && cell[0]==field.fields[0].box().hi[0];
  }
  auto prepare_external_trace_face_predicate() const {
    const int upper=field.fields.empty()?1:field.fields[0].box().hi[0];
    return [upper](int axis,int side,pops::Index<2> cell) {
      return axis==0 && side==1 && cell[0]==upper;
    };
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
    auto& fab=coverage.fields[0]; const auto nx=fab.box().extent()[0];
    for(int y=first;y<end;++y) set_host_value(fab,y*nx+nx-1,0);
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
  set_host_value(both.embedded.fields[0],1,0); emit(both,.01);
  count(selected(both).first,1); close(selected(both).second,-.003);
  // No local patches is legal; the global presence vote is another seam.
  Context empty(2,4,true); emit(empty,.01); count(empty.records.size(),0);
  // Uniform retains the prior no-coverage behavior.
  Context uniform(2,4); uniform.use_coverage=false; emit(uniform,.01);
  count(selected(uniform).first,4); close(selected(uniform).second,-.012);
  Context replica0(2,4),replica1(2,4);
  replica0.field.replicated=replica1.field.replicated=true;
  replica0.ranks=replica1.ranks=2; replica1.rank=1;
  emit(replica0,.01); emit(replica1,.01);
  count(selected(replica0).first,4); count(replica1.records.size(),0);
  close(selected(replica0).second+selected(replica1).second,-.012);
  Context owner1(2,4); owner1.rank=1; owner1.ranks=2;
  emit(owner1,.01); count(selected(owner1).first,4);
  close(selected(owner1).second,-.012);
  // A foreign mask cannot publish a partial batch.
  Context foreign(2,4); foreign.coverage.fields[0]=pops::Fab<2>(pops::Box<2>{pops::Index<2>{0,0},pops::Index<2>{0,3}},1);
  std::string diagnostic;
  try { emit(foreign,.01); } catch(const std::invalid_argument& e) { diagnostic=e.what(); }
  if(diagnostic.find("mask differs from local patches")==std::string::npos)
    throw std::runtime_error("foreign coverage was not rejected precisely");
  count(foreign.records.size(),0);
  std::cout << "transport coverage host PASS\n";
}
'''


def test_actual_transport_emitter_intersects_coverage_and_embedded_boundary(tmp_path):
    compiler, flags, link_flags = _host_toolchain()
    root = _sdk_root()
    generated = "\n".join(emit_transport_exchanges(
        "faces", "operation", "occurrence", "stage0/evaluation0", "dt"))
    source = tmp_path / "coverage.cpp"
    source.write_text(HOST.replace("@@EMITTER@@", generated))
    executable = tmp_path / "coverage"
    compiled = subprocess.run([compiler, "-std=c++20", "-O2", "-DPOPS_HAS_KOKKOS=1", "-I", str(root / "include"), *flags,
                    str(source), "-o", str(executable), *link_flags], check=False,
                   capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(executable)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "transport coverage host PASS\n"


def test_actual_diffusion_producer_intersects_coverage_and_embedded_boundary(tmp_path):
    """Execute the actual prepared producer; the scaffold supplies already evaluated faces.

    Diffusion retains incidence identities, rather than transport's exterior trace
    metadata. Select x+ by those exact identities and exercise its boundary-only path.
    This exercises native field storage and host publication, but not face evaluation,
    AMR preparation or MPI.
    """
    compiler, flags, link_flags = _host_toolchain()
    root = _sdk_root()
    header = (root / "include/pops/numerics/diffusion/prepared_diffusion.hpp").read_text()
    start = header.index("  template <class Context>\n  void stage_accepted_exchanges(")
    end = header.index("\n  const auto& faces()", start)
    producer = header[start:end].replace("Index<Dim>", "pops::Index<2>")
    scaffold = HOST.split("void emit(Context& ctx,double dt)")[0]
    # Transport faces store spatial integrals; prepared diffusion faces store flux density.
    scaffold = scaffold.replace("field<0>(),hy", "field<0>(),1").replace("field<1>(),.5", "field<1>(),1")
    source = tmp_path / "diffusion_coverage.cpp"
    source.write_text(scaffold + r'''
using pops::Real;
using pops::FieldView;
using pops::sync_host;
using pops::all_reduce_max;
using pops::runtime::program::accepted_exchange_contributes;
using pops::runtime::program::HostFacePublication;
using pops::runtime::program::HostFabPublication;
enum class DiffusiveBoundaryKind { periodic, prescribed };
struct Boundary { DiffusiveBoundaryKind kind=DiffusiveBoundaryKind::periodic; };
struct DiffusionGeometry {
  pops::Box<2> bounds; double hy;
  double spacing(int axis) const { return axis?hy:.5; }
  const pops::Box<2>& domain() const { return bounds; }
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
  pops::Lane lane{ctx.rank,ctx.ranks};
  const auto domain=ctx.field.local_size()?ctx.field.box(0):pops::Box<2>{};
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
  set_host_value(both.embedded.fields[0],1,0); emit(both,.01);
  count(selected(both).first,1); close(selected(both).second,.003);
  Context empty(2,4,true); emit(empty,.01); count(empty.records.size(),0);
  Context uniform(2,4); uniform.use_coverage=false; emit(uniform,.01);
  count(selected(uniform).first,4); close(selected(uniform).second,.012);
  Context replica0(2,4),replica1(2,4);
  replica0.field.replicated=replica1.field.replicated=true;
  replica0.ranks=replica1.ranks=2; replica1.rank=1;
  emit(replica0,.01); emit(replica1,.01);
  count(selected(replica0).first,4); count(replica1.records.size(),0);
  close(selected(replica0).second+selected(replica1).second,.012);
  Context owner1(2,4); owner1.rank=1; owner1.ranks=2;
  emit(owner1,.01); count(selected(owner1).first,4);
  close(selected(owner1).second,.012);
  Context foreign(2,4); foreign.coverage.fields[0]=pops::Fab<2>(pops::Box<2>{pops::Index<2>{0,0},pops::Index<2>{0,3}},1);
  std::string diagnostic;
  try { emit(foreign,.01); } catch(const std::invalid_argument& e) { diagnostic=e.what(); }
  if(diagnostic.find("mask differs from local patches")==std::string::npos)
    throw std::runtime_error("foreign diffusive coverage was not rejected precisely");
  count(foreign.records.size(),0);
  std::cout << "diffusion coverage host PASS\n";
}
''')
    executable = tmp_path / "diffusion_coverage"
    compiled = subprocess.run([compiler, "-std=c++20", "-O2", "-DPOPS_HAS_KOKKOS=1", "-I", str(root / "include"), *flags,
                               str(source), "-o", str(executable), *link_flags], check=False,
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
    compiler, flags, link_flags = _host_toolchain()
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
  context.coverage.fields[0]=pops::Fab<2>(pops::Box<2>{pops::Index<2>{0,0},pops::Index<2>{0,3}},1);
  try { probe.pointwise_exchange_coverage_mask(0,context.field); return 3; }
  catch(const std::invalid_argument&) {}
}
''')
    executable = tmp_path / "prepared_coverage"
    compiled = subprocess.run([compiler, "-std=c++20", "-O2", "-DPOPS_HAS_KOKKOS=1", "-I", str(root / "include"), *flags,
                               str(source), "-o", str(executable), *link_flags], check=False,
                              capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(executable)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
