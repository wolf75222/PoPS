"""Frozen snapshot bodies, producer phase guards and unchanged numerical emission."""
from pathlib import Path
import shutil
import subprocess

from pops.codegen.program_emit_transport_exchanges import emit_transport_exchanges

ROOT = Path(__file__).resolve().parents[2]
PIN = "61a3cb3fd175de8ee5a7bcfe6f5b1a007c2351bd"


def _body(source, signature):
    begin = source.index(signature)
    opening = source.index("{", begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def _frozen_method(name):
    path = "include/pops/runtime/program/" + name
    source = (ROOT / path).read_text()
    frozen = subprocess.run(["git", "show", PIN + ":" + path], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
    signature = "auto prepare_external_trace_face_predicate() const"
    method = _body(source, signature)
    assert method == _body(frozen, signature), "snapshot method drifted from declared SHA"
    return method


def test_actual_uniform_and_amr_snapshots_own_domain_and_topology(tmp_path):
    compiler = shutil.which("clang++")
    assert compiler, "independent host snapshot reception needs clang++; no masked skip"
    uniform = _frozen_method("program_context.hpp")
    amr = _frozen_method("amr_program_context.hpp")
    source = r'''
#include <array>
#include <iostream>
#include <stdexcept>
template<int Dim> using Index=std::array<int,Dim>;
enum class BoundarySide {lower,upper};
template<int Dim> struct Face {int axis;BoundarySide side;};
template<int Dim> struct Box {Index<Dim> lo{},hi{};};
template<int Dim> struct Geometry {
 Box<Dim> bounds; const auto& domain() const {return bounds;}
};
template<int Dim> struct Topology {
 std::array<bool,2*Dim> periodic{};
 bool is_periodic(Face<Dim> f) const {return periodic[2*f.axis+(f.side==BoundarySide::upper)];}
};
template<int Dim> struct Facade {
 Geometry<Dim> value;Topology<Dim> topology;
 mutable int geometries=0,topologies=0;bool forbidden=false;
 Geometry<Dim> geometry() const {
  if(forbidden) throw std::runtime_error("geometry callback inside producer");
  ++geometries;return value;
 }
 Topology<Dim> prepared_amr_boundary_topology() const {
  if(forbidden) throw std::runtime_error("topology callback inside producer");
  ++topologies;return topology;
 }
};
template<int Dim> struct Uniform {
 Facade<Dim>* facade_;
 auto geometry() const {return facade_->geometry();}
 auto scalar_boundary_topology_() const {return facade_->prepared_amr_boundary_topology();}
''' + uniform + r'''
};
template<int Dim> struct Amr {
 Facade<Dim>* facade_;
 auto geometry() const {return facade_->geometry();}
''' + amr + r'''
};
int checks=0;
void check(bool valid) {++checks;if(!valid) throw std::runtime_error("snapshot check failed");}
template<int Dim,template<int>class Context> void exercise() {
 Facade<Dim> facade;
 for(int axis=0;axis<Dim;++axis) {facade.value.bounds.lo[axis]=-2-axis;facade.value.bounds.hi[axis]=4+axis;}
 facade.topology.periodic[1]=true;
 Context<Dim> ctx{&facade};
 auto snapshot=ctx.prepare_external_trace_face_predicate();
 check(facade.geometries==1&&facade.topologies==1);
 const auto original=facade.value.bounds;
 facade.forbidden=true;facade.topology.periodic.fill(true);facade.value.bounds={};
 for(int axis=0;axis<Dim;++axis) {
  for(int side=0;side<2;++side) {
   auto cell=original.lo;cell[axis]=side?original.hi[axis]:original.lo[axis];
   check(snapshot(axis,side,cell)==!(axis==0&&side==1));
   cell[axis]+=side?-1:1;check(!snapshot(axis,side,cell));
  }
 }
 for(auto pair:std::array<std::array<int,2>,4>{{{-1,0},{Dim,1},{0,-1},{0,2}}}) {
  bool refused=false;try {snapshot(pair[0],pair[1],original.lo);}
  catch(const std::invalid_argument& e) {refused=std::string(e.what())=="external trace face has invalid axis or side";}
  check(refused);
 }
 check(facade.geometries==1&&facade.topologies==1);
 // Return the closure from a destroyed context/facade: no borrowed references.
 auto detached=[] {
  Facade<Dim> temporary;temporary.value.bounds.lo.fill(7);temporary.value.bounds.hi.fill(9);
  Context<Dim> local{&temporary};return local.prepare_external_trace_face_predicate();
 }();
 Index<Dim> cell{};cell.fill(9);check(detached(0,1,cell));
}
int main() {
 exercise<1,Uniform>();exercise<2,Uniform>();exercise<3,Uniform>();
 exercise<1,Amr>();exercise<2,Amr>();exercise<3,Amr>();
 std::cout<<"actual frozen snapshot checks="<<checks<<"\n";
}
'''
    cpp, binary = tmp_path / "snapshots.cpp", tmp_path / "snapshots"
    cpp.write_text(source)
    built = subprocess.run([compiler, "-std=c++20", "-O0", str(cpp), "-o", str(binary)],
                           capture_output=True, text=True, timeout=30)
    assert built.returncode == 0, built.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "actual frozen snapshot checks=90\n"


def test_geometry_acquisition_is_strictly_before_owner_and_producer():
    transport = "\n".join(emit_transport_exchanges("faces", "op", "occ", "stage", "dt"))
    producer = transport[transport.index("ctx.stage_exchange_batch("):]
    assert "ctx.geometry(" not in producer
    assert "ctx.is_external_trace_face(" not in producer
    assert "ctx.prepare_external_trace_face_predicate(" not in producer
    assert "accepted_geometry->spacing(tangent)" in producer
    assert "(*accepted_external_face)(axis, side, cell)" in producer
    assert transport.index("accepted_geometry.emplace") < transport.index("accepted_owner_contributes = pops::")
    assert transport.index("accepted_external_face.emplace") < transport.index("accepted_owner_contributes = pops::")
    assert transport.index("all_reduce_max(accepted_mask_layout_error") < transport.index("ctx.stage_exchange_batch(")
    diffusion = (ROOT / "include/pops/numerics/diffusion/prepared_diffusion.hpp").read_text()
    method = _body(diffusion, "void stage_accepted_exchanges(")
    producer = method[method.index("ctx.stage_exchange_batch("):]
    assert "ctx.geometry(" not in producer and "ctx.is_external_trace_face(" not in producer
    assert "ctx.prepare_external_trace_face_predicate(" not in producer
    assert "geometry_.spacing(tangent)" in producer
    assert "(*external_face)(axis, side, cell)" in producer
    assert method.index("external_face.emplace") < method.index("owner_contributes = accepted_exchange_contributes")
    assert method.index("all_reduce_max(active_layout_error") < method.index("ctx.stage_exchange_batch(")


def test_transport_and_diffusion_numerical_changes_are_only_snapshot_substitutions():
    path = "python/pops/codegen/program_emit_transport_exchanges.py"
    before = subprocess.run(["git", "show", PIN + "^:" + path], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
    after = (ROOT / path).read_text()
    additions = {
        '        "std::optional<pops::Geometry<pops::kNativeDimension>> accepted_geometry;",\n',
        '        "std::optional<decltype(ctx.prepare_external_trace_face_predicate())> accepted_external_face;",\n',
        '        "  accepted_geometry.emplace(ctx.geometry());",\n',
        '        "  accepted_external_face.emplace(ctx.prepare_external_trace_face_predicate());",\n',
    }
    for line in additions:
        assert after.count(line) == 1
        after = after.replace(line, "")
    after = after.replace("accepted_geometry->spacing(tangent)", "ctx.geometry().spacing(tangent)")
    after = after.replace("(*accepted_external_face)(axis, side, cell)", "ctx.is_external_trace_face(axis, side, cell)")
    assert before == after
    path = "include/pops/numerics/diffusion/prepared_diffusion.hpp"
    before = subprocess.run(["git", "show", PIN + "^:" + path], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
    after = (ROOT / path).read_text()
    for line in ("#include <optional>\n",
                 "    std::optional<decltype(ctx.prepare_external_trace_face_predicate())> external_face;\n",
                 "      external_face.emplace(ctx.prepare_external_trace_face_predicate());\n"):
        assert after.count(line) == 1
        after = after.replace(line, "")
    after = after.replace("(*external_face)(axis, side, cell)", "ctx.is_external_trace_face(axis, side, cell)")
    assert before == after
