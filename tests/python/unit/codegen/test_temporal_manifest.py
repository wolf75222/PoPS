"""Codegen temporal manifests carry authenticated persistent cache slots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from pops.codegen.temporal_manifest import build_temporal_manifest
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import AcceptedStep, Clock, Every, FixedDt, Hold, Schedule, TimePoint


@dataclass
class _Value:
    id: int
    clock: Clock
    point: TimePoint
    attrs: dict[str, Any] = field(default_factory=dict)
    op: str = "rhs"
    vtype: str = "scalar_field"
    block: Any = "owner"
    space: Any = "space"
    inputs: tuple[Any, ...] = ()
    state_ref: Any = None
    name: str = "scheduled"


class _Program:
    def __init__(self, value: _Value) -> None:
        self._values = (value,)
        self._histories = {}
        self._histories_ncomp = {}
        self._history_blocks = {}
        self._history_state_refs = {}
        self._history_spaces = {}
        self._history_contracts = {}
        self._history_persistence = {}
        self._time_states = {}
        self.clock = value.clock


def _manifest() -> dict[str, Any]:
    clock = Clock("macro")
    schedule = Schedule(Every(AcceptedStep(clock), 2), off=Hold())
    value = _Value(
        37,
        clock,
        TimePoint(clock),
        attrs={
            "schedule": schedule,
            "components": 1,
            "resolved_levels": (0, 1),
            "bytes": (8, 16),
            "maximum_bytes": (8, 16),
            "cells": (1, 2),
            "itemsize": 8,
        },
    )
    return build_temporal_manifest(_Program(value), target="amr_system")


def _fixed_run_payload() -> dict[str, Any]:
    return {"strategy": FixedDt(0.125).to_data(), "controls": {}}


def test_codegen_manifest_derives_all_cache_slots_from_complete_level_keys():
    manifest = _manifest()

    row = manifest["schedules"][0]
    assert row["node_id"] == 37
    assert row["cache_required"] is True
    assert row["cache_slots"] == [0, 1]
    assert [entry["key"]["level"] for entry in manifest["resource_plan"]["entries"]] == [0, 1]
    assert manifest["resource_plan_schema"] == "program-resource-plan:v1"
    assert len(manifest["resource_plan_digest"]) == 64

    state = TemporalRestartState()
    state.configure_program(manifest, time=0.0, macro_step=0)
    assert set(state.cache_cursors) == {"0", "1"}
    assert "37" not in state.cache_cursors


def test_codegen_manifest_rejects_cache_slot_not_owned_by_its_schedule():
    manifest = _manifest()
    manifest["schedules"][0]["cache_slots"] = [99]

    with pytest.raises(ValueError, match="absent from the ProgramResourcePlan"):
        TemporalRestartState().configure_program(manifest, time=0.0, macro_step=0)


def test_codegen_manifest_has_one_strict_schedule_row_schema():
    manifest = _manifest()
    manifest["schedules"][0].pop("cache_slots")

    with pytest.raises(ValueError, match="incomplete keys"):
        TemporalRestartState().configure_program(manifest, time=0.0, macro_step=0)


def test_slot_indexed_cache_cursors_survive_an_accepted_checkpoint_round_trip():
    manifest = _manifest()
    state = TemporalRestartState()
    state.configure_program(manifest, time=0.0, macro_step=0)
    state.begin_run(
        {"strategy": _fixed_run_payload(), "program_schedule": manifest},
        time=0.0,
        macro_step=0,
    )
    state.accept(before_time=0.0, before_step=0, time=0.125, macro_step=1)

    payload = state.checkpoint_json(time=0.125, macro_step=1)
    restored = TemporalRestartState.from_json(
        payload,
        time=0.125,
        macro_step=1,
        program_schedule=manifest,
    )
    assert restored.cache_cursors == state.cache_cursors
    assert set(restored.cache_cursors) == {"0", "1"}
