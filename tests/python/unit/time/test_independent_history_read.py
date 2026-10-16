"""Scientific ownership survives an issued output-history read and detachment."""

import pytest
import pops
from examples.api_v02.method_temporal_field import build_temporal_field
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_history_identity import (
    history_space_identity,
    OUTPUT_HISTORY_PROJECTION_SPACE,
)
from pops.time._program.detach import detach_compiled_program
from pops.time._program.global_history_reads import validate_read


def test_independent_history_read_keeps_exact_provider_through_real_lowering():
    spec = build_temporal_field(regrid_observation=True)
    p = spec.case._time
    retained = p.history("observed", lag=1)
    assert retained.block is None and retained.state_ref is None
    assert retained.attrs["field_authority"]["kind"] == "field_problem"
    assert validate_read(p, retained).block is p._history_blocks["observed"]
    assert history_space_identity(p, "observed") == OUTPUT_HISTORY_PROJECTION_SPACE
    p.sum(retained)
    cloned = detach_compiled_program(p)
    projected = next(
        v for v in cloned._values if v.op == "history" and v.attrs.get("global_field_history_read")
    )
    validate_read(cloned, projected)
    assert cloned._ir_hash() == p._ir_hash()
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    cpp = emit_cpp_program(
        resolved.time,
        target="amr_system",
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
    )
    assert 'ctx.history_observation("observed", 1, 1, 0,' in cpp
    assert "ctx.continuation_preflight_field(" in cpp
    assert "ctx.require_program_field_access(" in cpp


@pytest.mark.parametrize("change", ("owner", "provider", "width", "point"))
def test_independent_history_read_rejects_changed_scientific_authority(change):
    spec = build_temporal_field()
    p = spec.case._time
    retained = p.history("observed")
    if change == "owner":
        object.__setattr__(retained, "block", p._history_blocks["observed"])
    elif change == "point":
        from pops.time import TimePoint

        object.__setattr__(retained, "point", TimePoint(p.clock))
    else:
        attrs = dict(retained.attrs)
        if change == "width":
            attrs["ncomp"] = True
        else:
            attrs["field_authority"] = {**attrs["field_authority"], "unknown": "foreign"}
        object.__setattr__(retained, "attrs", attrs)
    with pytest.raises(ValueError, match="scientific authority"):
        validate_read(p, retained)


def test_independent_history_read_cannot_enlarge_declared_depth_or_borrow_state():
    spec = build_temporal_field()
    p = spec.case._time
    with pytest.raises(ValueError, match="maximum lag"):
        p.history("observed", lag=2)
    state = next(iter(p._time_states.values()))
    with pytest.raises(ValueError, match="State history"):
        p.history("observed", state_ref=state.state)


def test_uniform_independent_history_and_joint_continuation_builders_lower():
    from examples.api_v02.independent_field_history import build_independent_history
    from examples.api_v02.continuation_history_uniform import build_continuation_history_uniform

    for builder in (build_independent_history, build_continuation_history_uniform):
        spec = builder()
        resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
        cpp = emit_cpp_program(
            resolved.time,
            target="system",
            model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
        )
        assert "ctx.configure_history_field_metadata(" in cpp
        assert "ctx.store_history(" in cpp
        if builder is build_independent_history:
            assert "ctx.history_observation(" in cpp


def test_amr_independent_history_is_read_by_a_real_post_solve_consumer():
    spec = build_temporal_field(regrid_observation=True, retained_history_read=True)
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    cpp = emit_cpp_program(
        resolved.time,
        target="amr_system",
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
    )
    assert "ctx.history_observation(" in cpp
    assert "retained Field observation reduction" in cpp
    assert "ctx.configure_history_field_metadata(" in cpp
