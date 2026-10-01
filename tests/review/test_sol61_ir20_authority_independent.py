"""SOURCE_ONLY actual lease fragments; storage/authority/lane seams are substitutes."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "include/pops/runtime/program/prepared_amr_field_residual.hpp"


def braced(source, start):
    left = source.index("{", start)
    depth = 1
    right = left + 1
    while depth:
        depth += (source[right] == "{") - (source[right] == "}")
        right += 1
    return source[left + 1:right - 1], right


def public_request(budget=8192, order=(0, 1), *, emit=None):
    import pops
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D
    from pops.fields import (FieldProblem, FieldDiscretization, FieldBoundary, bcs,
                             CellCenteredNonlinearCoupled, FieldInteractionQuadrature,
                             CellVolumeMeasure, CellMidpoint, DirectSpatialInteraction,
                             SpatialInteractionKernel)
    from pops.model import Handle, OwnerPath
    from pops.math import Reaction, ValueExpr, SpatialInteraction
    from pops.solvers import Newton
    frame = CartesianDomain("independent-domain", lower=(-1., .2), upper=(2., 1.7)).frame(Cartesian2D())
    model = pops.Model("independent-load", frame=frame)
    load = model.state("rhs", components=("left", "right"))
    case = pops.Case("independent-nonlocal")
    block = case.block("captures", model, states=(load,))
    u, v = (Handle(n, kind="field", owner=OwnerPath.model("independent-fields")) for n in ("u", "v"))
    kernel = SpatialInteractionKernel(2, lambda x, y: 1 + 2*x[1] - 3*y[0] + x[0]*y[1])
    equations = (Reaction(u, 1 + ValueExpr(u)**2) == load[0],
                 Reaction(v, 1 + ValueExpr(v)**2) + SpatialInteraction(u, kernel, scale=2) == load[1])
    unknowns = (u, v)
    problem = FieldProblem("independent-original", unknowns=tuple(unknowns[i] for i in order),
        equations=tuple(equations[i] for i in order),
        boundaries=tuple(FieldBoundary(unknowns[i], bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())) for i in order))
    method = CellCenteredNonlinearCoupled(finite_difference_step=1e-7,
        interaction=FieldInteractionQuadrature(CellVolumeMeasure(), CellMidpoint(), DirectSpatialInteraction(budget)))
    field = case.field(problem, FieldDiscretization(method=method, boundaries=(), solver=Newton(tolerance=1e-10)))
    program = pops.Program("independent-program")
    state = program.state(block[load])
    request = field.bind_program_inputs(program=program, values={block[load]: state.n}, at=state.next.point,
                                       solver=field.default_program_solver())
    solved = program.solve(request, solver=field.default_program_solver())
    node = solved._token
    if emit is not None:
        from pops.time import FailRun, FixedDt
        from pops.initial import InitialCondition
        from pops.lib.initial import BindArray
        from pops.projection import ConservativeCellAverage
        from pops.layouts import Uniform, AMR
        from pops.mesh import CartesianGrid, PeriodicAxes
        from pops.codegen.program_codegen import emit_cpp_program
        from pops.codegen.program_models import ProgramModelGraph
        observed = field.observe(solved.consume(action=FailRun()))
        for identity in observed.unknowns:
            unknown = Handle.from_canonical_identity(identity)
            program.record_scalar(unknown.local_id, program.sum(observed[unknown]))
        program.commit(state.next, program.value("preserved-rhs", 1*state.n, at=state.next.point))
        program.step_strategy(FixedDt(.007))
        case.program(program)
        case.initials.add(InitialCondition(state=block[load], value=BindArray(), projection=ConservativeCellAverage()))
        grid = CartesianGrid(frame=frame, cells=(7, 5), periodic=PeriodicAxes(frame.axes))
        layout = Uniform(grid)
        if emit == "amr":
            from pops.amr import (AMRHierarchy, AMRExecution, AMRTransfer, AMRTagging,
                                  AMRRegrid, Tag, Buffer, Hysteresis, EqualityPolicy, ConflictPolicy)
            from pops.lib.amr import StateTransfer
            from pops.params import RuntimeParam
            from pops.time import every
            transfer = AMRTransfer()
            transfer.state(block[load], StateTransfer())
            threshold = case.param(RuntimeParam("marker", default=.5))
            layout = AMR(grid=grid, hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
                tagging=AMRTagging(rules=(Tag(ValueExpr(block[load])["left"] > case.value(threshold)), Buffer(cells=1)),
                    hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
                regrid=AMRRegrid(schedule=every(100, clock=program.clock)), transfer=transfer,
                execution=AMRExecution.synchronous())
        resolved = pops.resolve(pops.validate(case), layout=layout)
        code = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
                                target="amr_system" if emit == "amr" else "system")
        return code
    return program, node


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_actual_public_term_scope_and_mutations(order):
    from pops.fields._original_field_interaction import validate_interactions
    from pops.time._program.serialization import _json_ready
    program, node = public_request(order=order)
    assert program._serialize()["version"] == 20
    assert node.attrs["contract"] == "pops.spatial-field-residual@4"
    source = node.attrs["source_contract"]
    data = _json_ready(source["interactions"])
    term, = data["terms"]
    assert (term["row"], term["column"]) == (order.index(1), order.index(0))
    assert term["scale"] == {"kind": "integer", "value": "2"}
    assert node.attrs["solve_request"]["derivative"]["scheme"] == "central_full_residual"
    for key, wrong in (("row", True), ("column", order.index(1)), ("scale", {"kind": "integer", "value": "-2"})):
        forged = _json_ready(source["interactions"])
        forged["terms"][0][key] = wrong
        with pytest.raises((ValueError, TypeError)):
            validate_interactions(forged, source["field_problem"], source["unknown_components"])


def test_capture_retime_is_refused_by_actual_original_request_guard():
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
    program, node = public_request()
    capture = program._canonical_value(node.inputs[2])
    before = capture.point
    with pytest.raises(AttributeError, match="immutable"):
        capture.point = node.point
    try:
        object.__setattr__(capture, "point", node.point)
        assert capture.point != before
        with pytest.raises(ValueError, match="captures"):
            validate_nonlinear_field_request(program, node)
    finally:
        object.__setattr__(capture, "point", before)


@pytest.mark.parametrize("budget", [2**63 - 1, 2**63, 2**64 - 1])
def test_public_uint64_budget_reaches_real_emitter(budget):
    from pops.fields import FieldInteractionQuadrature, CellVolumeMeasure, CellMidpoint, DirectSpatialInteraction
    from pops.codegen.program_emit_original_interaction import emit_producer, interaction_terms
    encoded = FieldInteractionQuadrature(CellVolumeMeasure(), CellMidpoint(), DirectSpatialInteraction(budget)).to_data()
    assert encoded["max_workspace_bytes"] == (budget if budget < 2**63 else {"kind": "integer", "value": str(budget)})
    _, node = public_request(budget)
    lines = []
    emit_producer(node, "independent", interaction_terms(node), 0, lines, amr=True)
    assert f"{budget}ULL" in "\n".join(lines)


@pytest.mark.parametrize("invalid", [True, 0, -1, 2**64, 1.0, {"kind": "integer", "value": "1"},
                                     {"kind": "integer", "value": str(2**64)},
                                     {"kind": "real", "value": "0x1.0000000000000p+63"}])
def test_new_budget_codec_refuses_noncanonical_and_out_of_domain(invalid):
    from pops.fields._original_field_interaction import interaction_budget
    with pytest.raises((ValueError, TypeError)):
        interaction_budget({"max_workspace_bytes": invalid})


@pytest.mark.parametrize("backend", ["uniform", "amr"])
def test_real_public_resolved_emission_uses_candidate_and_permuted_route(backend):
    code = public_request(2**64 - 1, (1, 0), emit=backend)
    assert "18446744073709551615ULL" in code
    assert "interaction_0(index, 0)" in code
    assert "seal_original_field_source(" not in code
    assert "prepare_closed_original_interaction(" not in code
    assert "std::array<int, 1>{1}" in code
    if backend == "amr":
        assert "solve_interaction(" in code and "original_candidate_interaction(" in code
        assert "const auto& q, std::uint64_t evaluation" in code
    else:
        assert "ctx.spatial_interaction(" in code


def test_actual_native_lease_nonce_and_revocation(tmp_path):
    source = HEADER.read_text()
    require, _ = braced(source, source.index("void require_evaluation("))
    reentry, _ = braced(source, source.index("local_phase_(lane, [&]", source.index("SolveReport solve_impl_(")))
    # Only the actual lease phase; fake wrapper supplies explicit authority and lane seams.
    lease, _ = braced(source, source.index("\n      if (synchronized_interaction) {", source.index("auto evaluate =")))
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <functional>
using hierarchy_type = std::vector<double>;
struct ExecutionLane { int rank = 0; };
struct AmrFieldResidualAuthority { int epoch = 4; };
struct Provider {};
using provider_type = Provider;
struct Core {
  std::shared_ptr<Provider> provider_ = std::make_shared<Provider>();
  ExecutionLane prepared;
  const hierarchy_type* active_evaluation_ = nullptr;
  std::uint64_t active_evaluation_ordinal_ = 0, evaluation_sequence_ = 0;
  int votes = 0, authority_calls = 0;
  void require_authority(const AmrFieldResidualAuthority& a, const ExecutionLane& lane) const {
    if (&lane != &prepared || a.epoch != 4) throw std::logic_error("authority seam");
  }
  template<class F> void local_phase_(const ExecutionLane&, F f) const { f(); }
  void authenticate_(const hierarchy_type& q) const { if(q.size()!=3) throw std::logic_error("shape seam"); }
  void begin_again(){const auto& lane=prepared;local_phase_(lane,[&]{REENTRY});}
  void require_evaluation(const hierarchy_type& q, const AmrFieldResidualAuthority& current,
      const provider_type* expected_provider, std::uint64_t evaluation, const ExecutionLane& lane) const {
REQUIRE
  }
  void evaluate(const hierarchy_type& q, std::function<void(const hierarchy_type&, std::uint64_t)> producer,
                const AmrFieldResidualAuthority& current) {
    const auto& lane = prepared;
    const hierarchy_type* physical_q = &q;
    {
LEASE
    }
    assert(active_evaluation_ == nullptr); // actual branch ends before local physical body
  }
};
template<class F> void refuses(F f) { bool refused=false; try { f(); } catch(const std::logic_error&) { refused=true; } assert(refused); }
int main() {
  Core c; AmrFieldResidualAuthority auth; hierarchy_type q{1,2,3}, same_layout{1,2,3};
  ExecutionLane foreign; Provider foreign_provider;
  std::uint64_t previous=0; double produced=0;
  auto body = [&](const hierarchy_type& live, std::uint64_t nonce) {
    assert(&live == &q && nonce>previous);
    refuses([&]{c.begin_again();});
    c.require_evaluation(live,auth,c.provider_.get(),nonce,c.prepared);
    refuses([&]{c.require_evaluation(same_layout,auth,c.provider_.get(),nonce,c.prepared);});
    refuses([&]{c.require_evaluation(live,auth,&foreign_provider,nonce,c.prepared);});
    refuses([&]{c.require_evaluation(live,auth,c.provider_.get(),nonce,foreign);});
    refuses([&]{c.require_evaluation(live,{5},c.provider_.get(),nonce,c.prepared);});
    refuses([&]{c.require_evaluation(live,auth,c.provider_.get(),previous,c.prepared);});
    produced=live[0]+2*live[1]+3*live[2]; previous=nonce;
  };
  c.evaluate(q,body,auth); assert(produced==14 && previous==1);
  c.begin_again();
  refuses([&]{c.require_evaluation(q,auth,c.provider_.get(),previous,c.prepared);});
  q[0]=1.25; c.evaluate(q,body,auth); assert(produced==14.25 && previous==2);
  q[0]=.75; c.evaluate(q,body,auth); assert(produced==13.75 && previous==3);
  try { c.evaluate(q,[&](const auto&,auto){throw std::runtime_error("producer refusal");},auth); assert(false); }
  catch(const std::runtime_error&) {}
  assert(!c.active_evaluation_ && c.evaluation_sequence_==4);
  c.evaluate(q,body,auth); assert(previous==5); // retry does not rewind nonce
  c.evaluation_sequence_=std::numeric_limits<std::uint64_t>::max();
  bool overflow=false; try { c.evaluate(q,body,auth); } catch(const std::overflow_error&) {overflow=true;}
  assert(overflow && !c.active_evaluation_ && previous==5);
}
'''.replace("REQUIRE", require).replace("LEASE", lease).replace("REENTRY", reentry)
    path = tmp_path / "lease.cpp"
    path.write_text(cpp)
    executable = tmp_path / "lease"
    subprocess.run(["c++", "-std=c++20", "-fsanitize=undefined", "-fno-sanitize-recover=all", str(path), "-o", str(executable)], check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    # Actual lifecycle setup contains no sequence reset (checked over the whole definition).
    solve = source[source.index("SolveReport solve_impl_("):source.index("std::uint64_t evaluation_sequence_ = 0;")]
    assert "evaluation_sequence_ = 0" not in solve


@pytest.mark.parametrize("historical", [False, True])
def test_geometry_getter_exception_reaches_vote_before_peer_transport(tmp_path, historical):
    path = "include/pops/runtime/program/amr_program_context_spatial_interaction.inc"
    source = (subprocess.run(["git", "show", "0a5748aa6c19c9afa58ff566ee4dbe33671675a8:" + path],
                             cwd=ROOT, text=True, capture_output=True, check=True).stdout
              if historical else (ROOT / path).read_text())
    method, _ = braced(source, source.index("std::vector<field_type> original_candidate_interaction("))
    body, _ = braced(method, method.index("for (int level = 0; level < nlev(); ++level)"))
    cpp = r'''
#include <cassert>
#include <stdexcept>
#include <vector>
#include <exception>
struct Field{};
struct Level{const Field* field;int active,coverage,kappa,geometry;};
struct Facade{
  bool fail=false;
  int mask=0;
  int prepared_amr_level_geometry(int) {if(fail)throw std::out_of_range("live hierarchy seam");return 7;}
  int prepared_amr_block_level_active_mask(int,int){return 1;}
  const int& prepared_amr_block_level_coverage_mask(int,int){return mask;}
  int prepared_amr_block_level_volume_fraction(int,int){return 1;}
};
int votes=0; struct Lane{};
template<class F>void interaction_phase(const Lane&,F f){
  std::exception_ptr error;try{f();}catch(...){error=std::current_exception();}
  ++votes;if(error)std::rethrow_exception(error);
}
int main(){
  Lane lane;Facade facade;auto* facade_=&facade;int owner=0;int level=0;
  std::vector<Field> candidate(1);std::vector<Level> levels;
  auto run=[&]{BODY};
  run();assert(levels.size()==1 && votes==1);
  facade.fail=true;levels.clear();votes=0;
  bool refused=false;try{run();}catch(const std::out_of_range&){refused=true;}
  assert(refused && levels.empty());
  assert(votes==EXPECTED); // historical outside-vote counter remains distinguished
}
'''.replace("BODY", body).replace("EXPECTED", "0" if historical else "1").replace("&facade_->prepared_amr_block_level_coverage_mask", "facade_->prepared_amr_block_level_coverage_mask")
    path = tmp_path / "geometry.cpp"
    path.write_text(cpp)
    exe = tmp_path / "geometry"
    subprocess.run(["c++", "-std=c++20", str(path), "-o", str(exe)], check=True, capture_output=True, text=True)
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)
