"""Exact public Rate authority and versioned constitutive trace consumers, without JIT."""
import pops
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.support.integral_diffusion_case import build_diffusion_integral_case


def _emit(*, target="system", **options):
    case, layout, right, left, _ = build_diffusion_integral_case(**options)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    return resolved.time, right, left, emit_cpp_program(resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks), target=target)


@pytest.mark.parametrize("target,kind", (("system", "uniform"), ("amr_system", "amr2")))
def test_diffusive_trace_retains_exact_accepted_evaluation_and_version(target, kind):
    program, right, left, source = _emit(target=target, kind=kind)
    assert program._serialize()["version"] == 6
    assert len(program._integral_transfers) == 2
    deliveries = [line for line in source.splitlines() if "ctx.consume_external_trace(" in line]
    assert len(deliveries) == 2
    assert right.identity in deliveries[0] and left.identity in deliveries[1]
    assert all("diffusive-operation" in row and "/occurrence:" in row and "/evaluation:1" in row
               for row in deliveries)
    assert source.count(".stage_accepted_exchanges(") == 1
    if target == "system":
        assert source.index(deliveries[0]) < source.index("ctx.commit_many(")
    else:
        assert source.count("if (ctx.level() == 0) {") >= 1
        assert source.index("ctx.advance_hierarchy(dt") < source.rindex(".post_synchronization(dt)")


def test_unaccepted_diffusion_evaluation_cannot_supply_integral():
    with pytest.raises(ValueError, match="accepted, exact conservative face"):
        _emit(diagnostic=True)


def test_mixed_conservative_occurrences_require_an_exact_discriminator():
    from pops.codegen.program_emit_diffusion import _resolved_diffusive_trace_selection
    case, _, _, _, _ = build_diffusion_integral_case(mixed_transport=True)
    evaluation = next(value for value in case._time._values if value.op == "diffusive_rhs")
    with pytest.raises(ValueError, match="one exact conservative diffusive occurrence"):
        _resolved_diffusive_trace_selection(evaluation)


@pytest.mark.parametrize("kind", ("uniform", "amr2"))
def test_detached_diffusive_integral_relay_retains_exact_source_graph(kind):
    from pops.codegen._orchestration_compile import build_program_model_graph
    from pops.codegen.program_integral_transfers import emit_integral_transfers
    from pops.time._program.detach import detach_compiled_program

    case, layout, _, _, _ = build_diffusion_integral_case(kind)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = build_program_model_graph(resolved)
    detached = detach_compiled_program(resolved.time)
    assert detached._ir_hash() == resolved.time._ir_hash()
    variables = {value.id: "rate_%d" % value.id
                 for value in detached._values if value.op == "diffusive_rhs"}
    with pytest.raises(ValueError, match="detached accepted proof requires.*source authority"):
        emit_integral_transfers(detached, variables)
    expected = emit_integral_transfers(resolved.time, variables, model_authority=graph)
    actual = emit_integral_transfers(detached, variables, model_authority=graph)
    assert actual == expected
    assert len(actual) == 2
    original = emit_cpp_program(resolved.time, model_graph=graph,
                                field_plans=resolved.field_plans, target=resolved.target)
    compiled = emit_cpp_program(detached, model_graph=graph,
                                field_plans=resolved.field_plans, target=resolved.target)
    assert compiled == original

    # Another layout of the same physical Module is a valid source authority.
    # A distinct Module owner with a same-named State must remain foreign.
    from pops.layouts import Uniform
    grid = layout.mesh if kind == "uniform" else layout.grid
    foreign_model = pops.Model("foreign_diffusive_proof_owner", frame=grid.frame)
    foreign_state = foreign_model.state("U", components=("density",))
    foreign_case = pops.Case("foreign_diffusive_proof")
    foreign_block = foreign_case.block("fluid", foreign_model)
    foreign_program = pops.Program("foreign_identity")
    foreign_temporal = foreign_program.state(foreign_block[foreign_state])
    foreign_program.commit(foreign_temporal.next, foreign_program.value(
        "unchanged", foreign_temporal.n, at=foreign_temporal.next.point))
    foreign_case.program(foreign_program)
    foreign = pops.resolve(pops.validate(foreign_case), layout=Uniform(grid))
    with pytest.raises(ValueError):
        emit_integral_transfers(detached, variables,
                                model_authority=build_program_model_graph(foreign))


def test_production_diffusion_metadata_supplies_exact_consumable_ledger(tmp_path):
    """Extend the existing production producer host harness with the actual ledger.

    Only the harness main is extended. Its producer remains the extracted C++ method.
    The compile command uses the exact same source/SDK authority as the host harness.
    """
    from tests.python.unit.codegen import test_accepted_exchange_coverage_independent as host
    original_run = subprocess.run
    fragment = r'''
  using Ledger=pops::runtime::program::AcceptedExchangeLedger;
  Ledger ledger; ledger.declare_integral("q",.7);
  const auto initial=ledger.checkpoint();
  auto append=[&](const Context& context,int level,int substep) {
    pops::runtime::multiblock::BoundaryEvaluationPoint point;
    point.clock="host-ledger"; point.level=level; point.substep=substep;
    point.dt=level?.005:.01; point.physical_time=substep*point.dt;
    for(auto record:context.records) {
      record.qualify_runtime_point(point);
      ledger.stage(std::move(record));
    }
  };
  Context owned_coarse(2,4),owned_fine0(4,8),owned_fine1(4,8);
  owned_coarse.cover_right(0,4);
  emit(owned_coarse,.01); emit(owned_fine0,.005); emit(owned_fine1,.005);
  append(owned_coarse,0,0); append(owned_fine0,1,0); append(owned_fine1,1,1);
  const Ledger::TraceSelection selection{"operation","occurrence",0,1,0,"stage0/evaluation0"};
  const auto trace=ledger.prepare_trace(selection);
  if(trace.indices.size()!=16)
    throw std::runtime_error("diffusion trace metadata is absent from exact selector");
  close(trace.local_amount,.012);
  close(ledger.apply_trace("q",trace,trace.local_amount,1),.712);
  count(ledger.prepare_trace(selection).indices.size(),0);
  count(ledger.prepare_trace({"operation","occurrence",0,1,0,"other/evaluation"}).indices.size(),0);
  const auto accepted=ledger.checkpoint();
  try { ledger.apply_trace("q",trace,trace.local_amount,1); return 7; }
  catch(const std::invalid_argument&) {}
  if(ledger.checkpoint()!=accepted) return 8;
  const auto restored=Ledger::from_checkpoint(accepted);
  close(restored.integral("q"),.712);
  count(restored.prepare_trace(selection).indices.size(),0);
  ledger=Ledger::from_checkpoint(initial);
  close(ledger.integral("q"),.7); count(ledger.records().size(),0);
  append(owned_fine0,1,0); append(owned_fine1,1,1);
  const auto retry=ledger.prepare_trace(selection);
  close(ledger.apply_trace("q",retry,retry.local_amount,1),.712);
'''

    def compile_extended_main(command, *args, **kwargs):
        if "-std=c++20" in command:
            source = next(Path(arg) for arg in command if str(arg).endswith("diffusion_coverage.cpp"))
            text = source.read_text()
            marker = '  std::cout << "diffusion coverage host PASS\\n";'
            assert text.count(marker) == 1
            source.write_text(text.replace(marker, fragment + marker))
        return original_run(command, *args, **kwargs)

    with patch.object(host.subprocess, "run", compile_extended_main):
        host.test_actual_diffusion_producer_intersects_coverage_and_embedded_boundary(tmp_path)
