"""ADC-678 strict AMR accepted-state and topology checkpoint contracts."""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pops._generated_release_contract import (
    AMR_CHECKPOINT_PAYLOAD_VERSION,
    UNIFORM_CHECKPOINT_PAYLOAD_VERSION,
)
from pops.output._checkpoint_collective import restore_checkpoint_payload
from pops.runtime._amr_checkpoint_contract import (
    contract_for,
    encode_contract,
    preflight_contract,
    validate_restored_contract,
)
from pops.runtime._amr_checkpoint_v3 import (
    _checkpoint_amr_level_envelope,
    _live_amr_level_envelope,
    _require_exact_field_provider_depth,
)
from pops.runtime._amr_checkpoint_topology import (
    owner_ranks_for_boxes,
    recorded_rank_topology,
)
from pops.runtime._amr_system_io import _AmrSystemIO, _PreparedAMRSystemRestart
from pops.runtime._checkpoint_manifest import require_exact_payload_version


class _Payload(dict):
    @property
    def files(self):
        return list(self)


class _Sim:
    program_hash = "ab" * 32
    active_levels = 3
    configured_levels = 3

    def installed_program_hash(self):
        return self.program_hash

    def n_levels(self):
        return self.active_levels

    def configured_n_levels(self):
        return self.configured_levels

    def checkpoint_temporal_relations(self):
        return [[0, 1, 2, 1, "integral_only"], [1, 2, 3, 1, "integral_only"]]

    def checkpoint_transfer_routes(self):
        return [
            [
                "fluid.U",
                "prolong",
                "route.u",
                "provider.u",
                "kernel.linear",
                "cell.conservative",
                "cell",
                "conservative",
                "dense",
                "prolong",
                "2",
                "2,2",
                "2",
                "2",
            ]
        ]

    def program_accepted_state_manifest(self):
        return [
            [
                "rhs",
                "program.block.0",
                "fluid.U",
                "cell.conservative",
                "clock.macro",
                "dense.linear",
                "2",
                "3",
            ]
        ]

    def program_clock_manifest(self):
        return [["level", "0", "4", "0", "1", "0.4"], ["logical", "clock.macro", "4"]]

    def program_flux_ledger_manifest(self):
        return [
            [
                "program.block.0",
                "fluid.U",
                "rate.7",
                "physical_flux",
                "1",
                "4",
                "1",
                "2",
                "1",
                "2",
                "x_plus",
                "0.125",
                "0.05",
            ]
        ]

    def program_sync_manifest(self):
        return [
            ["0", "1", "0", "reflux", "4", "1", "1"],
            ["0", "1", "0", "average_down", "4", "1", "1"],
        ]


def _payload(sim=None):
    sim = sim or _Sim()
    return _Payload(
        {
            "amr_accepted_contract": np.array(encode_contract(sim)),
            "program_accepted_state": np.array([1, 2, 3], dtype=np.uint8),
            "regrid_count": np.array(4),
            "topology_epoch": np.array(7, dtype=np.uint64),
        }
    )


def test_contract_names_guarantee_relations_qualified_histories_and_transfer_plans():
    contract = contract_for(_Sim())
    assert contract["schema_version"] == 3
    assert contract["guarantee"] == "bit_identical_accepted_state"
    assert contract["ledger"]["accepted_entries"] == 1
    assert contract["ledger"]["transaction_depth"] == 0
    assert contract["ledger"]["entries"][0][8:10] == ["1", "2"]
    assert contract["level_relations"] == [
        {
            "parent": 0,
            "child": 1,
            "temporal_ratio": {"numerator": 2, "denominator": 1},
            "remainder_policy": "integral_only",
        },
        {
            "parent": 1,
            "child": 2,
            "temporal_ratio": {"numerator": 3, "denominator": 1},
            "remainder_policy": "integral_only",
        },
    ]
    assert contract["history_qualifications"][0][1:4] == [
        "program.block.0",
        "fluid.U",
        "cell.conservative",
    ]
    assert contract["transfer_routes"][0][2:5] == ["route.u", "provider.u", "kernel.linear"]
    assert contract["clocks"][1] == ["logical", "clock.macro", "4"]
    assert [row[3] for row in contract["synchronization"]] == ["reflux", "average_down"]


def test_preflight_returns_exact_native_payload_and_counters():
    state, regrids, epoch = preflight_contract(_Sim(), _payload())
    assert state == b"\x01\x02\x03"
    assert (regrids, epoch) == (4, 7)


@pytest.mark.parametrize("mutation", ["ratio", "route", "guarantee"])
def test_preflight_refuses_any_static_provenance_mismatch(mutation):
    payload = _payload()
    data = json.loads(str(payload["amr_accepted_contract"]))
    if mutation == "ratio":
        data["level_relations"][0]["temporal_ratio"]["numerator"] = 4
    elif mutation == "route":
        data["transfer_routes"][0][3] = "provider.other"
    else:
        data["guarantee"] = "regrid_on_restart"
    payload["amr_accepted_contract"] = np.array(json.dumps(data))
    with pytest.raises(ValueError, match="provenance differs"):
        preflight_contract(_Sim(), payload)


@pytest.mark.parametrize(
    "section", ["history_qualifications", "clocks", "ledger", "synchronization"]
)
def test_dynamic_contract_is_checked_after_the_opaque_state_is_restored(section):
    payload = _payload()
    data = json.loads(str(payload["amr_accepted_contract"]))
    if section == "history_qualifications":
        data[section][0][1] = "program.block.1"
    elif section == "ledger":
        data[section]["accepted_entries"] += 1
    else:
        data[section].append(["tampered"])
    payload["amr_accepted_contract"] = np.array(json.dumps(data))
    preflight_contract(_Sim(), payload)
    with pytest.raises(ValueError, match="restored AMR accepted-state image differs"):
        validate_restored_contract(_Sim(), payload)


def test_native_route_requires_no_program_blob_and_compiled_route_requires_one():
    compiled = _payload()
    compiled["program_accepted_state"] = np.array([], dtype=np.uint8)
    with pytest.raises(ValueError, match="requires a non-empty accepted state"):
        preflight_contract(_Sim(), compiled)

    native_sim = _Sim()
    native_sim.program_hash = ""
    native = _payload(native_sim)
    native["program_accepted_state"] = np.array([], dtype=np.uint8)
    assert preflight_contract(native_sim, native)[0] == b""


def test_checkpoint_level_envelope_accepts_derefined_active_depth():
    sim = _Sim()
    sim.active_levels = 1
    assert _live_amr_level_envelope(sim) == (1, 3)

    # A fresh runtime may still carry the complete configured hierarchy. Restart authenticates the
    # immutable envelope while allowing the checkpoint to rebuild its smaller accepted live depth.
    sim.active_levels = 3
    assert _checkpoint_amr_level_envelope(
        sim, {"n_levels": np.array(1), "configured_n_levels": np.array(3)}
    ) == (1, 3)


def test_checkpoint_level_envelope_refuses_a_different_configured_depth():
    sim = _Sim()
    with pytest.raises(ValueError, match="configured AMR depth"):
        _checkpoint_amr_level_envelope(
            sim, {"n_levels": np.array(1), "configured_n_levels": np.array(2)}
        )
    with pytest.raises(ValueError, match=r"outside its configured \[1, 3\]"):
        _checkpoint_amr_level_envelope(
            sim, {"n_levels": np.array(4), "configured_n_levels": np.array(3)}
        )


def test_incomplete_v5_level_envelope_is_an_offline_migration_input():
    class _FilesOnlyPayload:
        files = ("n_levels",)

        def __getitem__(self, key):
            if key != "n_levels":
                raise KeyError(key)
            return np.array(3)

    sim = _Sim()
    with pytest.raises(
        ValueError,
        match="configured_n_levels.*historical checkpoints require offline migration",
    ):
        _checkpoint_amr_level_envelope(sim, _FilesOnlyPayload())
    with pytest.raises(
        ValueError,
        match="configured_n_levels.*historical checkpoints require offline migration",
    ):
        _checkpoint_amr_level_envelope(sim, {"n_levels": np.array(1)})


@pytest.mark.parametrize(
    ("runtime_kind", "key", "expected"),
    [
        ("Uniform", "pops_checkpoint_version", UNIFORM_CHECKPOINT_PAYLOAD_VERSION),
        ("AMR", "pops_amr_checkpoint_version", AMR_CHECKPOINT_PAYLOAD_VERSION),
    ],
)
def test_uniform_and_amr_payload_versions_are_exact_current_integer_scalars(
    runtime_kind, key, expected,
):
    assert require_exact_payload_version(
        {key: np.array(expected, dtype=np.int64)},
        key=key,
        expected=expected,
        runtime_kind=runtime_kind,
    ) == expected

    for incompatible in (
        np.array(True),
        np.array(float(expected)),
        np.array(str(expected)),
        np.array([expected], dtype=np.int64),
    ):
        with pytest.raises(TypeError, match="exact integer scalar.*offline migration"):
            require_exact_payload_version(
                {key: incompatible},
                key=key,
                expected=expected,
                runtime_kind=runtime_kind,
            )

    with pytest.raises(ValueError, match="missing payload version.*offline migration"):
        require_exact_payload_version(
            {},
            key=key,
            expected=expected,
            runtime_kind=runtime_kind,
        )
    with pytest.raises(
        ValueError,
        match=rf"expected exactly {expected}.*offline migration",
    ):
        require_exact_payload_version(
            {key: np.array(expected - 1, dtype=np.int64)},
            key=key,
            expected=expected,
            runtime_kind=runtime_kind,
        )


@pytest.mark.parametrize(
    ("runtime_kind", "key", "expected"),
    [
        ("Uniform", "pops_checkpoint_version", UNIFORM_CHECKPOINT_PAYLOAD_VERSION),
        ("AMR", "pops_amr_checkpoint_version", AMR_CHECKPOINT_PAYLOAD_VERSION),
    ],
)
def test_historical_version_refusal_happens_before_restart_transaction(
    runtime_kind, key, expected,
):
    calls = []

    class _Executor:
        def _prepare_checkpoint_restart(self, _payload, *, bit_identical):
            calls.append("prepare")
            assert bit_identical is True
            require_exact_payload_version(
                {key: np.array(expected - 1, dtype=np.int64)},
                key=key,
                expected=expected,
                runtime_kind=runtime_kind,
            )

        def _begin_checkpoint_restart(self):
            calls.append("begin")

        def _apply_checkpoint_restart(self, _prepared):
            calls.append("apply")

        def _commit_checkpoint_restart(self):
            calls.append("commit")

        def _finalize_checkpoint_restart(self):
            calls.append("finalize")

        def _rollback_checkpoint_restart(self):
            calls.append("rollback")

    owner = SimpleNamespace(
        _execution_context=SimpleNamespace(
            communicator=SimpleNamespace(identity="serial", handle=None)
        )
    )
    with pytest.raises(ValueError, match="historical checkpoints require offline migration"):
        restore_checkpoint_payload(
            owner,
            _Executor(),
            b"historical",
            bit_identical=True,
        )
    assert calls == ["prepare"]


@pytest.mark.parametrize("phase", ["checkpoint capture", "restart"])
def test_field_provider_depth_must_equal_the_complete_active_hierarchy(phase):
    assert _require_exact_field_provider_depth("electric", 3, 3, phase=phase) is None
    with pytest.raises(ValueError, match=r"expected exactly the 3 active AMR levels"):
        _require_exact_field_provider_depth("electric", 2, 3, phase=phase)


def test_topology_owner_alignment_is_level_local_and_strict():
    payload = {"dmap_1": np.array([2, 0]), "dmap_2": np.array([1])}
    boxes = [(1, 0, 0, 3, 3), (1, 4, 4, 7, 7), (2, 2, 2, 5, 5)]
    assert owner_ranks_for_boxes(payload, boxes, 3) == [2, 0, 1]
    with pytest.raises(ValueError, match="truncated"):
        owner_ranks_for_boxes(payload, boxes + [(2, 6, 6, 7, 7)], 3)
    with pytest.raises(ValueError, match="lacks owner-rank map"):
        owner_ranks_for_boxes({}, boxes[:1], 3)


def _rank_topology_payload():
    return {
        "program_accepted_state_rank_0": np.array([1, 2], dtype=np.uint8),
        "program_accepted_state_rank_1": np.array([3, 4], dtype=np.uint8),
        "dmap_rank_0_level_0": np.array([0], dtype=np.int64),
        "dmap_rank_0_level_1": np.array([0, 1], dtype=np.int64),
        "dmap_rank_1_level_0": np.array([0], dtype=np.int64),
        "dmap_rank_1_level_1": np.array([0, 1], dtype=np.int64),
    }


def test_recorded_rank_topology_keeps_all_program_shards_and_one_exact_owner_map():
    topology = recorded_rank_topology(_rank_topology_payload(), 2, 2)
    assert topology.program_states == (b"\x01\x02", b"\x03\x04")
    assert topology.level_owner_ranks == ((0,), (0, 1))


def test_recorded_rank_topology_refuses_rank_local_owner_map_disagreement():
    payload = _rank_topology_payload()
    payload["dmap_rank_1_level_1"] = np.array([1, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="owner maps disagree"):
        recorded_rank_topology(payload, 2, 2)


def test_recorded_rank_topology_refuses_out_of_range_recorded_ownership():
    payload = _rank_topology_payload()
    payload["dmap_rank_0_level_1"] = np.array([0, 2], dtype=np.int64)
    payload["dmap_rank_1_level_1"] = np.array([0, 2], dtype=np.int64)
    with pytest.raises(ValueError, match=r"outside \[0, 2\)"):
        recorded_rank_topology(payload, 2, 2)


def test_recorded_rank_topology_refuses_mixed_program_presence():
    payload = _rank_topology_payload()
    payload["program_accepted_state_rank_1"] = np.array([], dtype=np.uint8)
    with pytest.raises(ValueError, match="disagree on whether a compiled Program image is present"):
        recorded_rank_topology(payload, 2, 2)


def test_rank_change_restart_rolls_back_after_hierarchy_rebuild_failure(monkeypatch):
    """The collective restart bracket must retain both native and Python snapshots through apply.

    The native ``AcceptedSnapshot`` already has direct C++ coverage for topology/history/flux
    restoration.  This source-only seam proves the new rank-change orchestration actually invokes
    that rollback after hierarchy replacement, rather than releasing the snapshot too early.
    """

    accepted_native = {
        "hierarchy": ((1, 0, 0, 7, 7),),
        "owners": ((0, 1),),
        "blocks": {"tracer": ((1.0, 2.0), (3.0, 4.0))},
        "aux": ((5.0, 6.0),),
        "potentials": ((7.0, 8.0),),
        "histories": {"rhs": ((9.0, 10.0), (11.0, 12.0))},
        "clock": (0.3, 3),
        "counters": (2, 5),
        "program_state": b"accepted-program-state",
        "program_revision": 7,
    }

    class _TransactionalNativeAMR:
        def __init__(self):
            self.state = deepcopy(accepted_native)
            self.snapshot = None
            self.rollback_calls = 0
            self.commit_calls = 0

        def begin_restart_transaction(self):
            if self.snapshot is not None:
                raise RuntimeError("nested test restart transaction")
            self.snapshot = deepcopy(self.state)

        def rebuild_hierarchy(self, boxes, owners):
            self.state["hierarchy"] = tuple(boxes)
            self.state["owners"] = tuple(owners)
            self.state["blocks"] = {"tracer": ((101.0,), (102.0,))}
            self.state["histories"] = {"rhs": ((103.0,), (104.0,))}
            self.state["program_state"] = b"partially-restored-program-state"
            self.state["program_revision"] = 8

        def commit_restart_transaction(self):
            self.commit_calls += 1
            self.snapshot = None

        def rollback_restart_transaction(self):
            self.rollback_calls += 1
            if self.snapshot is None:
                raise RuntimeError("test restart rollback lost its accepted snapshot")
            self.state = self.snapshot
            self.snapshot = None

    class _InjectedFailureAMR(_AmrSystemIO):
        def _prepare_checkpoint_restart(self, payload, *, bit_identical):
            assert payload == b"rank-change-checkpoint"
            assert bit_identical is False
            return _PreparedAMRSystemRestart("new-restart-identity", object())

    native = _TransactionalNativeAMR()
    runtime = _InjectedFailureAMR()
    runtime._s = native
    runtime._execution_context = SimpleNamespace(
        communicator=SimpleNamespace(identity="serial", handle=None)
    )
    runtime._last_restart_identity = "accepted-restart-identity"
    runtime._last_restart_report = "accepted-restart-report"
    runtime._temporal_restart_state = "accepted-temporal-state"
    runtime._step_controller = "accepted-step-controller"

    def fail_after_rebuild(owner, sim, prepared):
        assert owner is runtime
        assert sim is native
        sim.rebuild_hierarchy(
            ((1, 8, 8, 15, 15),),
            (0,),
        )
        sim.state["aux"] = ((201.0,),)
        sim.state["potentials"] = ((202.0,),)
        sim.state["clock"] = (0.8, 8)
        sim.state["counters"] = (9, 13)
        owner._last_restart_report = "partial-restart-report"
        owner._temporal_restart_state = "partial-temporal-state"
        owner._step_controller = None
        raise RuntimeError("injected failure after rank-change hierarchy rebuild")

    monkeypatch.setattr(
        "pops.runtime._amr_checkpoint_v3.apply_v3",
        fail_after_rebuild,
    )

    with pytest.raises(
        RuntimeError, match="injected failure after rank-change hierarchy rebuild"
    ):
        restore_checkpoint_payload(
            runtime,
            runtime,
            b"rank-change-checkpoint",
            bit_identical=False,
            phase_prefix="rank-change rollback proof",
        )

    assert native.state == accepted_native
    assert native.snapshot is None
    assert native.rollback_calls == 1
    assert native.commit_calls == 0
    assert runtime._last_restart_identity == "accepted-restart-identity"
    assert runtime._last_restart_report == "accepted-restart-report"
    assert runtime._temporal_restart_state == "accepted-temporal-state"
    assert runtime._step_controller == "accepted-step-controller"
    assert "_checkpoint_restart_python_snapshot" not in runtime.__dict__
    assert "_checkpoint_restart_committed" not in runtime.__dict__
