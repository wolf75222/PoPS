"""SOURCE_ONLY cap@2 reception; no Native/GTest/Case runtime qualification."""

import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def actual_body(text, signature):
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def test_actual_frozen_DSO_tuple_guard_refuses_each_foreign_projection(tmp_path):
    source = (ROOT / "include/pops/runtime/program/program_runtime_state.hpp").read_text()
    guard = actual_body(
        source, "const ProgramCheckpointHistoryMetadata& require_frozen_output_history_projection("
    )
    record = (
        actual_body(
            (ROOT / "include/pops/runtime/program/module_metadata.hpp").read_text(),
            "struct ProgramCheckpointHistoryMetadata",
        )
        + ";"
    )
    cpp = (
        r"""
#include <cassert>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
inline constexpr std::string_view kOutputHistoryProjectionSpace="pops.program.scalar-output-history-projection@2";
"""
        + record
        + r"""
struct Runtime {
 bool artifact_backed_=true;
 std::string installed_hash_="frozen-program";
 std::vector<int> block_map_{4,5};
 struct Metadata {std::vector<ProgramCheckpointHistoryMetadata> histories;} checkpoint_metadata_;
"""
        + guard
        + r"""
};
int main() {
 int checked=0;
 for (int depth:{2,3,9,257}) for(int width:{1,3,5}) {
 Runtime r;
 r.checkpoint_metadata_.histories.push_back({"ring",1,"exact-issued-observation","pops.program.scalar-output-history-projection@2","clock","interp",depth,width});
 auto call=[&](Runtime& v){return v.require_frozen_output_history_projection("ring",5,"exact-issued-observation","pops.program.scalar-output-history-projection@2","clock","interp",depth,width,1);};
 assert(call(r)==r.checkpoint_metadata_.histories.front()); ++checked;
 std::vector<std::function<void(Runtime&)>> attacks{
  [](auto& v){v.artifact_backed_=false;}, [](auto& v){v.installed_hash_.clear();},
  [](auto& v){v.block_map_[1]=4;}, [](auto& v){v.block_map_.resize(1);},
  [](auto& v){v.checkpoint_metadata_.histories.clear();},
  [](auto& v){v.checkpoint_metadata_.histories.push_back(v.checkpoint_metadata_.histories.front());},
  [](auto& v){v.checkpoint_metadata_.histories[0].name="other";},
  [](auto& v){v.checkpoint_metadata_.histories[0].program_owner=0;},
  [](auto& v){v.checkpoint_metadata_.histories[0].state_identity="scalar-history:ring";},
  [](auto& v){v.checkpoint_metadata_.histories[0].space_identity="scalar-output-field-v1";},
  [](auto& v){v.checkpoint_metadata_.histories[0].clock_identity="other";},
  [](auto& v){v.checkpoint_metadata_.histories[0].interpolation_identity="other";},
  [](auto& v){++v.checkpoint_metadata_.histories[0].depth;},
  [](auto& v){++v.checkpoint_metadata_.histories[0].components;}};
 for(const auto& attack:attacks) {
  Runtime wrong=r;attack(wrong); bool refused=false;
  try{call(wrong);}catch(const std::invalid_argument&){refused=true;}
  assert(refused);++checked;
 }
 }
 assert(checked==180);
}
"""
    )
    path = tmp_path / "actual_frozen_tuple_guard.cpp"
    path.write_text(cpp)
    executable = tmp_path / "actual_frozen_tuple_guard"
    subprocess.run(
        ["rtk", "proxy", "c++", "-std=c++20", "-O0", str(path), "-o", str(executable)], check=True
    )
    subprocess.run(["rtk", "proxy", str(executable)], check=True)


@pytest.mark.parametrize("depth", (1, 3, 8))
def test_real_IR16_global_store_gets_separate_output_capability(depth):
    from pops.codegen.program_history_identity import (
        history_space_identity,
        OUTPUT_HISTORY_PROJECTION_SPACE,
    )
    from pops.time._program.global_history_storage import descriptor

    path = ROOT / "tests/review/test_sol61_history_storage_owner_received.py"
    spec = importlib.util.spec_from_file_location("independent_capability_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, program, blocks, _, problem, observation, field, _ = module._baseline()
    value = observation[field[problem.unknowns[2]]]
    inputs = tuple(value.inputs[0].inputs)
    program.store_history("global-output", value, depth=depth, owner_block=blocks[0])
    before = program._serialize()
    assert history_space_identity(program, "global-output") == OUTPUT_HISTORY_PROJECTION_SPACE
    assert descriptor(program, "global-output") != "scalar-history:global-output"
    assert program._serialize() == before
    assert value.block is value.space is value.state_ref is None
    assert all(a is b for a, b in zip(inputs, value.inputs[0].inputs, strict=True))


def test_remap_issued_metadata_precedes_transfer_and_no_Q_cold_promotion():
    text = (ROOT / "src/runtime/amr/amr_system.cpp").read_text()
    marker = text.index("// Validate the complete artifact-issued tuple")
    remaining = text[marker:]
    guard = remaining.index("require_frozen_output_history_projection(")
    transfer = remaining.index(
        "AMR output history projection supports exact aligned 1:1 spatial remap only"
    )
    assert guard < transfer
    assert "ratio.numerator != 1 || ratio.denominator != 1" in remaining
    assert "validate_history_sample_provenance(samples[slot], initialized, dts[slot])" in remaining
    assert "std::bit_cast<std::array<std::byte, sizeof(Real)>>(a)" in remaining
    public = (
        ROOT / "include/pops/runtime/program/amr_program_context_history_checkpoint_public.inc"
    ).read_text()
    assert "HistorySampleIdentity::zero_start()" in public
    assert "runtime_state().require_frozen_output_history_projection(" in public
    services = (
        ROOT / "include/pops/runtime/program/amr_program_context_history_checkpoint_services.inc"
    ).read_text()
    assert "manager.space_identity.at(key) == kOutputHistoryProjectionSpace" in services
    assert "not Program reads" in services


def test_real_AMR_case_metadata_and_all_registration_sites_use_same_frozen_tuple():
    import importlib.util
    import json
    import pops
    from pops.amr import (
        AMRExecution,
        AMRHierarchy,
        AMRRegrid,
        AMRTagging,
        AMRTransfer,
        Buffer,
        ConflictPolicy,
        EqualityPolicy,
        Hysteresis,
        Tag,
    )
    from pops.analytic import x
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.initial import InitialCondition
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.lib.initial import Analytic
    from pops.math import ValueExpr
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.params import RuntimeParam
    from pops.projection import ConservativeCellAverage
    from pops.time import FixedDt, every
    from pops.time._program.global_history_storage import descriptor

    source = ROOT / "tests/review/test_sol61_history_storage_owner_received.py"
    spec = importlib.util.spec_from_file_location("independent_real_capability_case", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case, program, blocks, states, problem, observation, field, _ = module._baseline()
    value = observation[field[problem.unknowns[2]]]
    program.store_history("actual-global-output", value, depth=3, owner_block=blocks[0])
    transfer = AMRTransfer()
    for index, (block, state) in enumerate(zip(blocks, states, strict=True)):
        current = program.state(block[state])
        program.commit(
            current.next,
            program.value("actual-physical-%d" % index, 1 * current.n, at=current.next.point),
        )
        frame = block._instance_registry._blocks[block.local_id]["model"].frame
        case.initials.add(
            InitialCondition(
                state=block[state],
                value=Analytic(frame=frame, components=tuple(0 * x(frame) for _ in state)),
                projection=ConservativeCellAverage(),
            )
        )
        transfer.state(block[state], StateTransfer())
    program.step_strategy(FixedDt(0.125))
    case.program(program)
    threshold = case.param(RuntimeParam("actual-cap-threshold", default=0.5))
    layout = AMR(
        grid=CartesianGrid(frame=frame, cells=(3, 5), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(blocks[0][states[0]])[0] > case.value(threshold)),
                Buffer(cells=0),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(1000, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
    )
    plan = pops.resolve(pops.validate(case), layout=layout)
    cpp = emit_cpp_program(
        plan.time, model=ProgramModelGraph.from_resolved_blocks(plan.blocks), target="amr_system"
    )
    assert plan.time._serialize()["version"] == 16
    assert len(plan.initial_condition_plan.bindings) == 3
    expected = json.dumps(descriptor(plan.time, "actual-global-output"))
    registrations = [
        line for line in cpp.splitlines() if 'ctx.register_history("actual-global-output"' in line
    ]
    assert len(registrations) >= 2
    assert all(
        expected in line and "pops.program.scalar-output-history-projection@2" in line
        for line in registrations
    )
    shape = cpp.split("pops_program_checkpoint_history_state_identity", 1)[1]
    assert expected in shape
