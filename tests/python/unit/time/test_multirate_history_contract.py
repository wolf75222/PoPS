"""ADC-667: typed histories and explicit cross-clock history providers."""
from dataclasses import dataclass
from fractions import Fraction

import pytest

from typed_program_support import state_refs, typed_state

from pops.identity.semantic import program_semantic_data, semantic_identity_of
from pops.time import (
    Clock,
    DenseOutput,
    HistoryContract,
    HistoryValidity,
    InterpolateHistory,
    LinearInterpolation,
    NoInterpolation,
    Program,
    ProgramGraph,
    SampleAndHold,
    Synchronize,
    TimePoint,
)


def test_keep_history_builds_complete_owner_qualified_contract():
    program = Program("history_contract")
    state = typed_state(program, "fluid", state_name="U")
    program.keep_history(state, depth=3, interpolation=LinearInterpolation())

    contract = state.prev.contract
    assert type(contract) is HistoryContract
    assert contract.owner == program.owner_path.canonical()
    assert contract.state is state.state
    assert contract.space == state.space
    assert contract.clock is program.clock
    assert contract.validity == HistoryValidity(
        TimePoint(program.clock, step=-3), TimePoint(program.clock))
    assert contract.interpolation.to_data()["kind"] == "linear"
    assert state.prev.validity is contract.validity
    assert state.prev.interpolation is contract.interpolation


def test_default_history_is_exact_samples_only():
    program = Program("exact_history")
    state = typed_state(program, "fluid", state_name="U")
    program.keep_history(state, depth=2)

    assert state.prev.contract.interpolation.to_data()["kind"] \
        == NoInterpolation().to_data()["kind"]
    with pytest.raises(ValueError, match="interpolation=.*capability"):
        InterpolateHistory(state.prev)


def test_history_contract_snapshots_extension_capability_data():
    class Extension:
        __pops_history_interpolation__ = True

        def __init__(self):
            self.order = 3

        def to_data(self):
            return {"kind": "extension_dense", "schema_version": 1, "order": self.order}

    capability = Extension()
    program = Program("extension")
    state = typed_state(program, "fluid", state_name="U")
    program.keep_history(state, depth=2, interpolation=capability)
    capability.order = 99

    assert state.prev.contract.interpolation.to_data()["order"] == 3


def test_history_interpolation_is_an_explicit_cross_clock_provider():
    program = Program("multirate")
    state = typed_state(program, "fluid", state_name="U")
    program.keep_history(state, depth=2, interpolation=DenseOutput(order=2))
    fast = Clock("fast", owner=program.owner_path)

    synchronized = program.synchronize(
        state.prev,
        at=TimePoint(fast),
        relation=InterpolateHistory(state.prev),
        name="U_fast",
    )
    relation = synchronized.attrs["relation"]
    assert relation["kind"] == "history_interpolation"
    assert relation["provider"]["kind"] == "typed_history"
    provider_contract = relation["provider"]["contract"]
    assert provider_contract["state"]["qualified_id"] \
        == state.prev.contract.to_data()["state"]["qualified_id"]
    assert provider_contract["interpolation"]["kind"] == "dense_output"
    assert program.to_graph().nodes


def test_history_provider_must_match_the_transferred_history():
    program = Program("mismatch")
    first = typed_state(program, "a", state_name="U")
    second = typed_state(program, "b", state_name="U")
    program.keep_history(first, depth=2, interpolation=LinearInterpolation())
    program.keep_history(second, depth=2, interpolation=LinearInterpolation())
    fast = Clock("fast", owner=program.owner_path)

    with pytest.raises(ValueError, match="same HistoryHandle"):
        program.synchronize(
            first.prev,
            at=TimePoint(fast),
            relation=InterpolateHistory(second.prev),
        )


@dataclass(frozen=True)
class _ProviderlessRelation:
    __pops_sync_relation__ = True

    def validate_transfer(self, source, target):
        del source, target

    def to_data(self):
        return {"kind": "providerless_extension", "schema_version": 1}


def test_cross_clock_extension_without_provider_is_rejected():
    program = Program("providerless")
    state = typed_state(program, "fluid", state_name="U")
    fast = Clock("fast", owner=program.owner_path)

    with pytest.raises(ValueError, match="explicit provider"):
        program.synchronize(
            state.n, at=TimePoint(fast), relation=_ProviderlessRelation())


def test_history_contract_is_in_program_graph_restart_manifest_and_identity():
    def authored(capability):
        program = Program("identity")
        state = typed_state(program, "fluid", state_name="U")
        program.keep_history(state, depth=2, interpolation=capability)
        _ = state.prev.value
        return program

    linear = authored(LinearInterpolation())
    dense = authored(DenseOutput(2))
    assert linear._ir_hash() != dense._ir_hash()

    graph_data = linear.to_graph().to_data()
    history = next(node for node in graph_data["nodes"] if node.get("op") == "history")
    assert history["attrs"]["attrs"]["history_contract"]["interpolation"]["kind"] == "linear"

    manifest_history = linear.temporal_manifest()["histories"][0]
    assert manifest_history["interpolation"]["kind"] == "linear"
    assert manifest_history["validity"]["oldest"]["step"] == -2
    assert manifest_history["validity"]["newest"]["step"] == 0


def test_keep_history_contract_is_in_the_compiler_semantic_identity():
    program = Program("semantic_identity")
    state = typed_state(program, "fluid", state_name="U")
    program.keep_history(state, depth=2, interpolation=LinearInterpolation())
    renamed = Program("same_semantics_different_name")
    renamed_state = typed_state(renamed, "fluid", state_name="U")
    renamed.keep_history(
        renamed_state, depth=2, interpolation=LinearInterpolation())
    dense = Program("semantic_identity")
    dense_state = typed_state(dense, "fluid", state_name="U")
    dense.keep_history(dense_state, depth=2, interpolation=DenseOutput(2))

    semantic = program_semantic_data(program)
    assert semantic["history_contracts"][0]["interpolation"]["kind"] == "linear"
    identity = semantic_identity_of(program=program)
    assert identity.domain == "semantic"
    assert identity == semantic_identity_of(program=renamed)
    assert identity != semantic_identity_of(program=dense)


def test_program_graph_rejects_a_provider_for_a_different_history():
    program = Program("graph_provider_binding")
    state = typed_state(program, "fluid", state_name="U")
    program.keep_history(state, depth=2, interpolation=LinearInterpolation())
    fast = Clock("fast", owner=program.owner_path)
    program.synchronize(
        state.prev,
        at=TimePoint(fast),
        relation=InterpolateHistory(state.prev),
    )
    graph = program.to_graph()
    synchronization = graph.nodes[-1]
    relation = synchronization.relation.to_data()
    relation["provider"]["contract"]["state"]["qualified_id"] = "forged-state"
    forged = Synchronize(
        synchronization.node_id,
        synchronization.value,
        synchronization.source_clock,
        synchronization.target_clock,
        relation,
        synchronization.point,
        name=synchronization.name,
    )

    with pytest.raises(ValueError, match="does not match its source history"):
        ProgramGraph(
            graph.name,
            (*graph.nodes[:-1], forged),
            clocks=graph.clocks,
        )


def _interpolated_program(capability):
    program = Program("native_history_interpolation")
    state = typed_state(program, "fluid", state_name="U")
    program.keep_history(state, depth=2, interpolation=capability)
    fast = Clock("fast", owner=program.owner_path)
    interpolated = program.synchronize(
        state.prev,
        at=TimePoint(fast, step=-1),
        relation=InterpolateHistory(state.prev),
    )
    advanced = program.subcycle(
        interpolated,
        clock=fast,
        within=program.clock,
        count=2,
        body_fn=lambda P, value: P.value("fast_copy", 1 * value),
    )
    returned = program.synchronize(
        advanced,
        at=state.next.point,
        relation=SampleAndHold(),
    )
    program.commit(state.next, returned)
    return program


def _child_owned_interpolated_program():
    program = Program("native_child_history_interpolation")
    block, declared = state_refs(program, "fluid", state_name="U")
    macro = program.state(block[declared])
    fast = Clock("fast", owner=program.owner_path)
    child = program.state(block[declared], clock=fast)
    program.keep_history(child, depth=2, interpolation=LinearInterpolation())
    advanced = program.subcycle(
        child.n,
        clock=fast,
        within=program.clock,
        count=2,
        body_fn=lambda P, value: P.value("child_copy", 1 * value),
    )
    # The source ring is owned and rotated by the child clock. The fractional macro coordinate is
    # -3/4 macro ticks == -3/2 child ticks, so native lowering must use adjacent child samples.
    program.synchronize(
        child.prev,
        at=TimePoint(program.clock, Fraction(1, 4), step=-1),
        relation=InterpolateHistory(child.prev),
        name="child_history_at_macro_coordinate",
    )
    returned = program.synchronize(
        advanced,
        at=macro.next.point,
        relation=SampleAndHold(),
    )
    program.commit(macro.next, returned)
    return program


def test_linear_history_interpolation_lowers_to_the_native_uniform_slot_kernel():
    from pops.codegen.program_codegen import emit_cpp_program

    source = emit_cpp_program(
        _interpolated_program(LinearInterpolation()),
        model=None,
        target="system",
    )

    assert "ctx.interpolate_history_linear" in source
    assert "ctx.scratch_state" in source


def test_child_clock_history_interpolation_lowers_with_its_qualified_clock_ledger():
    from pops.codegen.program_codegen import emit_cpp_program

    program = _child_owned_interpolated_program()
    source = emit_cpp_program(program, model=None, target="system")
    schedule = program.temporal_manifest()
    history, = schedule["histories"]
    child_clock = history["clock"]

    assert child_clock != program.clock.qualified_id
    assert "ctx.store_history" in source
    assert 'ctx.rotate_histories("%s")' % child_clock in source
    assert "ctx.interpolate_history_linear" in source


def test_dense_and_amr_history_interpolation_lowering_fail_closed():
    from pops.codegen.program_codegen import emit_cpp_program

    with pytest.raises(NotImplementedError, match="supported capability"):
        emit_cpp_program(_interpolated_program(DenseOutput(2)), model=None)
    with pytest.raises(NotImplementedError, match="Uniform-only"):
        emit_cpp_program(
            _interpolated_program(LinearInterpolation()),
            model=None,
            target="amr_system",
        )


def test_validity_interval_cannot_mix_clocks_or_run_backwards():
    slow = Clock("slow")
    fast = Clock("fast")
    with pytest.raises(ValueError, match="same clock"):
        HistoryValidity(TimePoint(slow, step=-1), TimePoint(fast))
    with pytest.raises(ValueError, match="must not follow"):
        HistoryValidity(TimePoint(slow, step=1), TimePoint(slow))
