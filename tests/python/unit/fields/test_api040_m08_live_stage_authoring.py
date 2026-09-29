"""Public M08 regression: a field solve consumes the current SSPRK2 state.

This deliberately imports only Python source and performs validate/resolve;
native acceptance remains the installed runner's responsibility.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXAMPLES = ROOT / "examples/migration/scientific"


def test_two_heterogeneous_blocks_publish_the_current_stage_field(monkeypatch):
    monkeypatch.setenv("POPS_API040_M08_AUTHORING_ONLY", "1")
    monkeypatch.syspath_prepend(str(ROOT / "python"))
    monkeypatch.syspath_prepend(str(EXAMPLES))
    authored = runpy.run_path(str(EXAMPLES / "api040_m08_guiding_center.py"))
    initial = authored["initial_fields"].field_context
    predictor = authored["predictor_fields"].field_context
    assert len(initial.stage_sources) == len(predictor.stage_sources) == 2
    assert initial.field == predictor.field
    assert dict(initial.stage_sources)[authored["charge_block"]] == authored["q0"].id
    assert dict(initial.stage_sources)[authored["tracer_block"]] == authored["c0"].id
    assert dict(predictor.stage_sources)[authored["charge_block"]] == authored["q1"].id
    assert dict(predictor.stage_sources)[authored["tracer_block"]] == authored["c1"].id
    assert initial != predictor
    assert authored["initial_point"].time == authored["probe_point"].time
    assert authored["probe_q"].id != authored["q0"].id
    solves = [value for value in authored["program"]._values if value.op == "solve_linear"]
    identities = [value.attrs["solve_request"]["equation_identity"] for value in solves]
    assert len(identities) == len(set(identities)) == 3
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    emitted = emit_cpp_program(authored["program"], model_graph=
        ProgramModelGraph.from_resolved_blocks(authored["resolved"].blocks))
    assert emitted.count("ctx.publish_field_components(") == 2
    assert "apply_general_field<pops::kNativeDimension" in emitted
