"""Exact checkpoint import ordering and real retained RHS budget controls."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pops.codegen.program_emit_amr import _flux_expression_budgets
from pops.numerics.terms import DefaultSource
from pops.runtime._system_io_history import restore_histories
from pops.time import Program
from pops.time._history.persistence import Interval
from tests.python.support.typed_program import program_states, synthetic_module


def _state(program):
    _, states = program_states(program, synthetic_module(program.name + "_model", components=("u",)),
                               ("blk",))
    return states["blk"]


@pytest.mark.parametrize("lag", [2, 4])
def test_pure_state_history_has_no_phantom_flux_basis(lag):
    program = Program("pure_state_%d" % lag)
    state = _state(program).n
    program.store_history("state", state, depth=lag)
    program.history("state", lag=lag, space=state.space, block=state.block, state_ref=state.state_ref)
    assert _flux_expression_budgets(program) == ((0, 0),)


@pytest.mark.parametrize("lag,expected", [(1, 2), (2, 3)])
def test_real_source_rhs_history_retains_current_and_lagged_bases(lag, expected):
    program = Program("rhs_history_%d" % lag)
    state = _state(program).n
    rhs = program.rhs(state=state, terms=[DefaultSource()])
    program.store_history("rate", rhs, depth=lag)
    program.history("rate", lag=lag, space=state.space, block=state.block, state_ref=state.state_ref)
    assert _flux_expression_budgets(program) == ((expected, 1),)


def test_branch_result_with_source_rhs_preserves_retained_allowance():
    program = Program("branch_rhs_history")
    state = _state(program).n
    selected = program.branch(
        program.norm2(state) > 0,
        lambda branch: branch.rhs(state=state, terms=[DefaultSource()]),
        lambda branch: branch.rhs(state=state, terms=[DefaultSource()]),
    )
    program.store_history("rate", selected, depth=1)
    program.history("rate", lag=1, space=state.space, block=state.block, state_ref=state.state_ref)
    assert _flux_expression_budgets(program) == ((2, 1),)


def _payload():
    payload = {"history_names": ("first", "second"), "macro_step": 8, "regrid_every": 0}
    for name in payload["history_names"]:
        payload.update({
            "history_depth_" + name: 3,
            "history_levels_" + name: (0, 1),
            "history_policy_" + name: json.dumps(Interval(2).to_manifest()),
            "history_requested_stored_slots_" + name: (0, 2),
            "history_stored_slots_" + name: (0, 2),
            "history_storage_mode_" + name: "policy",
        })
        for level in (0, 1):
            suffix = "%s_level_%d" % (name, level)
            payload["history_init_" + suffix] = True
            payload["history_fill_count_" + suffix] = 3
            payload["history_slot_dt_" + suffix] = np.full(3, 0.002 / (2 ** level))
            for slot in (0, 2):
                payload["history_%s_%d" % (suffix, slot)] = np.asarray([level + slot + 1.0])
    return payload


class _RecordingRuntime:
    def __init__(self):
        self.events = []

    def history_levels(self, name):
        return (0, 1)

    def restore_history(self, name, level, slot, values):
        self.events.append(("anchor", name, level, slot))

    def restore_history_metadata(self, *args):
        raise AssertionError("atomic provenance must own the full metadata publication")

    def restore_history_provenance(self, name, level, dt, initialized, fill_count):
        assert initialized and fill_count == 3
        assert np.array_equal(dt, np.full(3, 0.002 / (2 ** level)))
        self.events.append(("provenance", name, level))

    def rebuild_history_slots(self, name, anchors):
        assert anchors == [0, 2]
        self.events.append(("replay", name))
        return 1


def test_authenticated_import_follows_all_anchors_and_precedes_every_replay():
    runtime = _RecordingRuntime()

    def import_accepted():
        assert len([row for row in runtime.events if row[0] == "anchor"]) == 8
        assert len([row for row in runtime.events if row[0] == "provenance"]) == 4
        assert not any(row[0] == "replay" for row in runtime.events)
        runtime.events.append(("accepted_import",))

    report = restore_histories(runtime, _payload(), before_replay=import_accepted)
    assert runtime.events[-3:] == [("accepted_import",), ("replay", "first"), ("replay", "second")]
    assert all(row["recomputed_slots"] == 1 for row in report.histories)


def test_failed_authenticated_import_never_executes_replay():
    runtime = _RecordingRuntime()

    def reject():
        raise ValueError("checkpoint history authority differs")

    with pytest.raises(ValueError, match="history authority differs"):
        restore_histories(runtime, _payload(), before_replay=reject)
    assert not any(row[0] == "replay" for row in runtime.events)


def test_invalid_import_callback_refuses_before_history_mutation():
    runtime = _RecordingRuntime()
    with pytest.raises(TypeError, match="before_replay must be callable"):
        restore_histories(runtime, _payload(), before_replay=False)
    assert runtime.events == []
