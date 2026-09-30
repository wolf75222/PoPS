"""Independent host reception of 61a3cb3 plus 9471bf8; no native/JIT/MPI execution.

Reuse only the existing thin field/face adapter. Execute the actual transport
emission and actual PreparedDiffusion method with separate adversarial assertions.
The vote/batch spies measure local control flow, not MPI synchronization.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from pops.codegen.program_emit_transport_exchanges import emit_transport_exchanges
from tests.python.unit.codegen.test_accepted_exchange_coverage_independent import HOST


ROOT = Path(__file__).resolve().parents[2]


def _scaffold():
    scaffold = HOST.split("void emit(Context& ctx,double dt)")[0]
    scaffold = scaffold.replace(
        "inline long all_reduce_max(long value,const Lane&) { return value; }",
        "inline int votes=0;\n"
        "inline long all_reduce_max(long value,const Lane&) { ++votes; return value; }")
    scaffold = scaffold.replace("int rank=0, ranks=1;",
                                "int rank=0, ranks=1; int batches=0, completed=0; bool fail_active=false;\n"
                                "mutable int geometry_calls=0,predicate_calls=0;\n"
                                "bool in_producer=false,fail_geometry=false,fail_predicate=false;")
    scaffold = scaffold.replace("Geometry geometry() const { return {hy}; }",
                                "Geometry geometry() const {\n"
                                "  if(in_producer) throw std::runtime_error(\"collective geometry inside producer\");\n"
                                "  ++geometry_calls;\n"
                                "  if(fail_geometry) throw std::runtime_error(\"injected geometry preparation\");\n"
                                "  return {hy}; }")
    scaffold = scaffold.replace("auto prepare_external_trace_face_predicate() const {",
                                "auto prepare_external_trace_face_predicate() const {\n"
                                "  if(in_producer) throw std::runtime_error(\"topology inside producer\");\n"
                                "  ++predicate_calls;\n"
                                "  if(fail_predicate) throw std::runtime_error(\"injected topology preparation\");")
    scaffold = scaffold.replace("bool is_external_trace_face(int axis,int side,pops::Index cell) const {",
                                "bool is_external_trace_face(int axis,int side,pops::Index cell) const {\n"
                                "  if(in_producer) throw std::runtime_error(\"facade trace callback inside producer\");")
    scaffold = scaffold.replace("return use_embedded?&embedded:nullptr;",
                                'if(fail_active) throw std::runtime_error("injected active preparation");\n'
                                "    return use_embedded?&embedded:nullptr;")
    scaffold = scaffold.replace("auto candidate=records;",
                                "++batches; auto candidate=records; in_producer=true;")
    scaffold = scaffold.replace("records.swap(candidate);",
                                "in_producer=false; records.swap(candidate); ++completed;")
    return scaffold


MAIN = r'''
void check(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
void close(double a,double b) { check(std::abs(a-b)<2e-15,"wrong independent amount"); }
std::pair<int,double> selected(const Context& ctx,int component) {
  int n=0; double amount=0;
  for(const auto& r:ctx.records)
    if(r.exterior_trace && r.trace_axis==0 && r.trace_side==1 && r.trace_component==component) {
      ++n; amount+=r.integrated_amount();
    }
  return {n,amount};
}
void run(Context& ctx) {
  pops::votes=0; const int batches=ctx.batches, completed=ctx.completed;
  const int predicates=ctx.predicate_calls, geometries=ctx.geometry_calls;
  emit(ctx,.01);
  check(pops::votes==2,"owner/nonowner missed a mask vote");
  check(ctx.batches==batches+1 && ctx.completed==completed+1,"empty producer missed batch tail");
  check(ctx.predicate_calls==predicates+1,"nonowner/empty missed topology snapshot");
  check(ctx.geometry_calls==geometries+GEOMETRY_CALLS,"nonowner/empty missed geometry snapshot");
}
void refuse(Context& ctx,int votes) {
  const auto prior=ctx.records.size(); const int batches=ctx.batches;
  pops::votes=0;
  try { emit(ctx,.01); throw std::runtime_error("foreign mask accepted"); }
  catch(const std::invalid_argument&) {}
  check(pops::votes==votes,"failure skipped or added a mask vote");
  check(ctx.batches==batches && ctx.records.size()==prior,"refusal changed prior prefix");
}
int main() {
  // Seven replicas still represent exactly one physical trace, all components.
  double amount[2]={0,0}; int records[2]={0,0};
  for(int rank=0;rank<7;++rank) {
    Context c(2,4); c.rank=rank;c.ranks=7;c.field.replicated=true;
    c.cover_right(2,4); c.use_embedded=true; c.embedded.fields[0].data[1]=0;
    run(c);
    for(int component=0;component<2;++component) {
      const auto [n,a]=selected(c,component); records[component]+=n;amount[component]+=a;
      if(rank) check(c.records.empty(),"replica published an interior or exterior record");
    }
  }
  check(records[0]==1 && records[1]==1,"EB/finest-owner intersection duplicated");
  close(amount[0],SIGN*.003);close(amount[1],SIGN*.006);
  // Distributed nonzero lane ordinals must contribute; rank zero may be empty.
  Context empty(2,4,true);empty.ranks=7;run(empty);check(empty.records.empty(),"empty rank emitted");
  Context distributed(2,4);distributed.rank=6;distributed.ranks=7;run(distributed);
  check(selected(distributed,0).first==4,"distributed rank6 incorrectly suppressed");
  close(selected(distributed,0).second,SIGN*.012);close(selected(distributed,1).second,SIGN*.024);
  // A suppressed replica still authenticates every mask before any batch write.
  Context bad(2,4);bad.rank=6;bad.ranks=7;bad.field.replicated=true;
  bad.records=distributed.records;
  bad.coverage.fields[0].bounds.hi[0]=0;refuse(bad,2);
  Context bad_eb(2,4);bad_eb.rank=1;bad_eb.ranks=2;bad_eb.field.replicated=true;
  bad_eb.use_embedded=true;bad_eb.embedded.fields[0].bounds.lo[0]=1;refuse(bad_eb,2);
  Context lookup(2,4);lookup.rank=1;lookup.ranks=2;lookup.field.replicated=true;
  lookup.fail_active=true;refuse(lookup,1);
  Context topology(2,4);topology.rank=1;topology.ranks=2;topology.field.replicated=true;
  topology.fail_predicate=true;refuse(topology,1);
  Context empty_bad(2,4,true);empty_bad.rank=6;empty_bad.ranks=7;
  empty_bad.coverage.fields=distributed.coverage.fields;refuse(empty_bad,2);
  // No-coverage Uniform and fully covered AMR retain their independent meaning.
  Context uniform(2,4);uniform.rank=6;uniform.ranks=7;uniform.use_coverage=false;run(uniform);
  close(selected(uniform,0).second,SIGN*.012);
  Context covered(2,4);covered.cover_right(0,4);run(covered);check(selected(covered,0).first==0,"covered face emitted");
  // Empty replicated producer preserves an already staged accepted prefix.
  distributed.field.replicated=true;const auto prior=distributed.records.size();
  const auto saved=distributed.records.front();run(distributed);
  check(distributed.records.size()==prior && distributed.records.front().key()==saved.key(),"empty batch damaged prefix");
  std::cout << "independent replicated exchange host PASS\n";
}
'''


def _source(kind, revision=None):
    scaffold = _scaffold()
    if kind == "transport":
        emitter = emit_transport_exchanges
        if revision is not None:
            path = "python/pops/codegen/program_emit_transport_exchanges.py"
            source = subprocess.run(["git", "show", revision + ":" + path], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout
            namespace = {}
            exec(compile(source, path, "exec"), namespace)
            emitter = namespace["emit_transport_exchanges"]
        body = "\n".join(emitter("faces", "op", "occ", "evaluation", "dt"))
        return scaffold + "void emit(Context& ctx,double dt) {const auto& faces=ctx.faces;\n" + body + "\n}\n" + MAIN.replace("SIGN", "-1").replace("GEOMETRY_CALLS", "1")
    path = "include/pops/numerics/diffusion/prepared_diffusion.hpp"
    header = (ROOT / path).read_text() if revision is None else subprocess.run(
        ["git", "show", revision + ":" + path], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout
    start = header.index("  template <class Context>\n  void stage_accepted_exchanges(")
    end = header.index("\n  const auto& faces()", start)
    body = header[start:end].replace("Index<Dim>", "pops::Index")
    scaffold = scaffold.replace("FaceAxis{hy},FaceAxis{.5}", "FaceAxis{1},FaceAxis{1}")
    return scaffold + r'''
using pops::Real;using pops::FieldView;using pops::sync_host;using pops::all_reduce_max;
using pops::runtime::program::accepted_exchange_contributes;
enum class DiffusiveBoundaryKind { periodic,prescribed };
struct Boundary {DiffusiveBoundaryKind kind=DiffusiveBoundaryKind::periodic;};
struct Geometry {pops::Box bounds;double hy;
  double spacing(int axis) const {return axis?hy:.5;}
  const pops::Box& domain() const {return bounds;}};
struct Producer {static constexpr int Dim=2,Components=2;using Field=pops::MultiFab<2>;
  const Field& variable_;const std::vector<Faces>& faces_;Geometry geometry_;
  std::array<Boundary,4> physical_{};const pops::Lane* lane_;
  Real explicit_frequency() const {return 0;}
''' + body + r'''
};
void emit(Context& ctx,double dt) {
  pops::Lane lane{ctx.rank,ctx.ranks};
  const auto box=ctx.field.local_size()?ctx.field.box(0):pops::Box{};
  Producer producer{ctx.field,ctx.faces,{box,ctx.hy},{},&lane};
  producer.physical_[1].kind=DiffusiveBoundaryKind::prescribed;
  producer.stage_accepted_exchanges(ctx,0,"op","occ","evaluation",dt,true);
}
''' + MAIN.replace("SIGN", "1").replace("GEOMETRY_CALLS", "0")


@pytest.mark.parametrize("kind", ["transport", "diffusion"])
def test_actual_producer_preserves_nonowner_votes_and_refusals(tmp_path, kind):
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "independent host reception requires a C++20 compiler; no masked skip"
    source, binary = tmp_path / f"{kind}.cpp", tmp_path / kind
    source.write_text(_source(kind))
    compiled = subprocess.run([compiler, "-std=c++20", "-O1", "-I", str(ROOT / "include"),
                               str(source), "-o", str(binary)], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "independent replicated exchange host PASS\n"


@pytest.mark.parametrize("kind", ["transport", "diffusion"])
def test_old_collective_getters_are_rejected_by_the_same_phase_guard(tmp_path, kind):
    """A positive owner baseline actually visits faces; old producer getters must trip."""
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "phase guard reception requires C++20; no masked skip"
    source = _source(kind, "61a3cb3fd175de8ee5a7bcfe6f5b1a007c2351bd^")
    source = source.replace("int main() {", "int reception_main() {")
    source += '\nint main(){try{return reception_main();}catch(const std::exception& e){std::cout<<e.what();return 42;}}\n'
    cpp, binary = tmp_path / "old.cpp", tmp_path / "old"
    cpp.write_text(source)
    built = subprocess.run([compiler, "-std=c++20", "-O1", "-I", str(ROOT / "include"),
                            str(cpp), "-o", str(binary)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 42
    assert result.stdout == ("collective geometry inside producer" if kind == "transport"
                             else "facade trace callback inside producer")


def test_actual_batch_collectives_and_other_producer_contracts():
    helper = (ROOT / "include/pops/runtime/program/accepted_exchange.hpp").read_text()
    prepare = helper[helper.index("std::vector<ExchangeRecord> prepare_exchange_batch("):]
    assert prepare.index("collective_step_rejection_phase(") < prepare.index("std::forward<Producer>(producer)")
    staging = prepare[prepare.index("inline void stage_exchange_batch_collectively("):]
    assert staging.index("for (auto& record : records)") < staging.index("collectively_rethrow_exception(")
    assert "ledger.restore_size(prior_size);" in staging
    for context_name, authority in [("program_context.hpp", "system_"),
                                    ("amr_program_context.hpp", "facade_")]:
        context = (ROOT / "include/pops/runtime/program" / context_name).read_text()
        start = context.index("  void stage_exchange_batch(Producer&& producer) const {")
        end = context.index("\n  }", start)
        batch = context[start:end]
        assert "prepare_exchange_batch(" in batch
        assert f"{authority}->stage_program_exchanges(records);" in batch
        assert "if (" not in batch and "return" not in batch
    moving = (ROOT / "include/pops/runtime/program/program_context_moving_interval.inc").read_text()
    codec = (ROOT / "include/pops/runtime/program/moving_interval_checkpoint.hpp").read_text()
    # Moving source/geometry/amount records support a rank-local codec contract.
    assert "accepted_exchange_contributes" not in moving
    assert '"receipt lacks its exact accepted exchange occurrence"' in codec
    assert "actual.exterior_trace==expected.exterior_trace" in codec
    # Joint inventory amounts already use collective reductions, no trace support.
    inventory = (ROOT / "python/pops/codegen/program_interaction_exchanges.py").read_text()
    assert "ctx.sum_component(" in inventory and "ctx.stage_exchange(" in inventory
    assert "exterior_trace" not in inventory
