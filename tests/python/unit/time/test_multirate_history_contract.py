"""ADC-667: typed histories and explicit cross-clock history providers."""
from dataclasses import dataclass

import pytest

from typed_program_support import typed_state

from pops.time import (
    Clock,
    DenseOutput,
    HistoryContract,
    HistoryValidity,
    InterpolateHistory,
    LinearInterpolation,
    NoInterpolation,
    Program,
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

    assert state.prev.contract.interpolation.to_data()["kind"] == NoInterpolation().to_data()["kind"]
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

    assert state.prev.contract.interpolation.to_data()["order"]["scalar"]["value"] == "3"


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
    assert provider_contract["state"]["qualified_id"] == state.prev.contract.to_data()["state"]["qualified_id"]
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


def test_history_contract_is_part_of_program_identity_and_detached_snapshot():
    def authored(capability):
        program = Program("identity")
        state = typed_state(program, "fluid", state_name="U")
        program.keep_history(state, depth=2, interpolation=capability)
        _ = state.prev.value
        return program

    linear = authored(LinearInterpolation())
    dense = authored(DenseOutput(2))
    assert linear._ir_hash() != dense._ir_hash()

    detached = linear.to_graph()
    data = detached.to_data()
    history = next(node for node in data["nodes"] if node.get("op") == "history")
    assert history["attrs"]["attrs"]["history_contract"]["interpolation"]["kind"] == "linear"


def test_validity_interval_cannot_mix_clocks_or_run_backwards():
    slow = Clock("slow")
    fast = Clock("fast")
    with pytest.raises(ValueError, match="same clock"):
        HistoryValidity(TimePoint(slow, step=-1), TimePoint(fast))
    with pytest.raises(ValueError, match="must not follow"):
        HistoryValidity(TimePoint(slow, step=1), TimePoint(slow))
