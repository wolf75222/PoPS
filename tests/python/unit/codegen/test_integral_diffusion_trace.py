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
