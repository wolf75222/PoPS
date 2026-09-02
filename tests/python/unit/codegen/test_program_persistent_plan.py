"""Deterministic compiler lowering for Program persistent resources."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

import pops.codegen.program_persistent_plan as plan_mod
from pops.codegen.program_codegen import (
    _emit_candidate_tables,
    _emit_system_install,
    _validate_resource_slot_calls,
)
from pops.codegen.program_emit_ops import _append_generated_field_route_preparation
from pops.codegen.program_emit_amr import _emit_amr_candidate_entry_suffix
from pops.codegen.program_persistent_plan import (
    ProgramPersistentValueKey,
    ProgramResourcePlan,
    ProgramResourcePlanEntry,
    lower_program_resource_plan,
)
from pops.output._checkpoint_contract import (
    ProgramPersistentValueCheckpoint,
    encode_program_persistent_value_checkpoint,
    program_persistent_value_checkpoint_capacity,
)


@dataclass(frozen=True)
class _Value:
    id: int
    op: str = "scratch"
    vtype: str = "scalar_field"
    attrs: dict[str, Any] = field(default_factory=lambda: {"components": 1, "bytes": 8})
    block: Any = "owner"
    space: Any = "space"
    clock: Any = "clock"
    inputs: tuple[Any, ...] = ()
    point: Any = None
    state_ref: Any = None
    name: str = "value"


class _Program:
    def __init__(self, *values: _Value) -> None:
        self._values = tuple(values)


def _entry(value_id: int, path: str = "root/0", **kwargs: Any) -> ProgramResourcePlanEntry:
    key = ProgramPersistentValueKey(
        value_id=value_id,
        occurrence_path=path,
        owner="owner",
        space="space",
        clock="clock",
    )
    return ProgramResourcePlanEntry(key=key, bytes=8, **kwargs)


def test_checkpoint_capacity_uses_the_neutral_nominal_plan_authority() -> None:
    plan = ProgramResourcePlan((_entry(1),))
    assert program_persistent_value_checkpoint_capacity(plan) > plan.maximum_bytes

    class StructuralImpostor:
        digest = plan.digest
        maximum_bytes = plan.maximum_bytes

        @staticmethod
        def abi_rows():
            return plan.abi_rows()

    with pytest.raises(TypeError, match="exact sealed ProgramResourcePlan"):
        program_persistent_value_checkpoint_capacity(StructuralImpostor())


def test_occurrences_are_preorder_and_nested_paths_are_canonical() -> None:
    child = _Value(3, name="child")
    branch = _Value(
        9,
        op="branch",
        attrs={
            "components": 1,
            "bytes": 8,
            "true_block": (child,),
            "false_block": (_Value(4, name="other"),),
        },
    )
    occurrences = list(plan_mod.iter_program_occurrences(_Program(branch)))
    assert [path for _value, path in occurrences] == [
        "root/0",
        "root/0/true_block/0",
        "root/0/false_block/0",
    ]


def test_local_newton_residual_path_omits_only_its_inline_storage_occurrence() -> None:
    """The same storage operation remains materialized outside ``residual_block``.

    Program values are canonical single-occurrence objects, so this legal graph
    uses two otherwise equivalent ``source`` occurrences.  The decision must be
    made from the static path: the inline residual expression gets no slot,
    while the outer source retains its dense prepared-storage row.
    """

    inline_source = _Value(3, op="source", name="inline-source")
    residual_branch = _Value(
        2,
        op="branch",
        attrs={"true_block": (inline_source,), "false_block": ()},
        name="inline-residual-branch",
    )
    local_newton = _Value(
        1,
        op="solve_local_nonlinear",
        attrs={
            "components": 1,
            "bytes": 8,
            "residual_block": (residual_branch,),
        },
        name="local-newton",
    )
    outer_source = _Value(4, op="source", name="outer-source")

    plan = lower_program_resource_plan(_Program(local_newton, outer_source))

    assert [(row.slot, row.key.value_id, row.key.occurrence_path) for row in plan] == [
        (0, 1, "root/0"),
        (1, 4, "root/1"),
    ]
    assert plan.slot_for_value(outer_source) == 1
    with pytest.raises(KeyError, match="needs its full occurrence path"):
        plan.slot_for_value(inline_source)


def test_complete_key_disambiguates_owner_space_clock_and_level() -> None:
    common = dict(value_id=7, occurrence_path="root/0")
    keys = {
        ProgramPersistentValueKey(**common, owner="a", space="s", clock="c", level=None),
        ProgramPersistentValueKey(**common, owner="b", space="s", clock="c", level=None),
        ProgramPersistentValueKey(**common, owner="a", space="t", clock="c", level=None),
        ProgramPersistentValueKey(**common, owner="a", space="s", clock="d", level=2),
    }
    assert len(keys) == 4


def test_plan_is_stable_sorted_and_dense_with_authenticated_digest() -> None:
    values = (_entry(8, "root/2"), _entry(2, "root/0"), _entry(5, "root/1"))
    first = ProgramResourcePlan(values)
    second = ProgramResourcePlan(tuple(reversed(values)))
    assert [(row.key.value_id, row.slot) for row in first] == [(2, 0), (5, 1), (8, 2)]
    assert first.digest == second.digest
    assert first.to_json() == second.to_json()
    with pytest.raises(AttributeError, match="immutable"):
        first.maximum_bytes = 1


def test_abi_rows_are_lossless_versioned_and_rehydratable() -> None:
    entry = _entry(
        8,
        "root/2",
        shape=(4, 5),
        cells=20,
        itemsize=8,
        lifetime="persistent_schedule",
        off_policy="hold",
        transfer_provider="amr.regrid.v1",
        restart_provider="checkpoint.v2",
        components=1,
        ghosts=2,
        maximum_bytes=256,
        communicates=True,
        restart_required=True,
    )
    plan = ProgramResourcePlan((entry,))
    row = plan.abi_rows()[0]
    assert row.schema == "program-resource-plan:v1"
    assert row.plan_digest == plan.digest
    assert row.slot == 0
    assert row.key.occurrence_path == "root/2"
    assert row.transfer_provider == "amr.regrid.v1"
    assert row.restart_provider == "checkpoint.v2"
    assert row.bytes == 8 and row.maximum_bytes == 256
    assert row.shape == (4, 5) and row.cells == 20 and row.itemsize == 8
    assert row.to_data()["key"]["occurrence_path"] == "root/2"
    assert plan.abi_data()["digest"] == plan.digest
    assert plan.abi_data()["entries"][0]["slot"] == 0
    restored = ProgramResourcePlan.from_data(plan.to_data())
    assert restored.to_json() == plan.to_json()


def test_rehydration_rejects_tampered_path_digest() -> None:
    data = ProgramResourcePlan((_entry(1),)).to_data()
    data["entries"][0]["key"]["occurrence_path_id"] += 1
    with pytest.raises(ValueError, match="unauthenticated"):
        ProgramResourcePlan.from_data(data)


def test_duplicate_collision_and_overflow_are_refused() -> None:
    duplicate = _entry(1)
    with pytest.raises(ValueError, match="duplicate"):
        ProgramResourcePlan((duplicate, duplicate))

    original_path_digest = plan_mod.occurrence_path_digest
    plan_mod.occurrence_path_digest = lambda _path: 17
    try:
        with pytest.raises(ValueError, match="digest collision"):
            ProgramResourcePlan((_entry(1, "root/a"), _entry(2, "root/b")))
    finally:
        plan_mod.occurrence_path_digest = original_path_digest

    original_identity_digest = plan_mod._identity_digest
    plan_mod._identity_digest = lambda _payload: "collision"
    try:
        with pytest.raises(ValueError, match="identity digest collision"):
            ProgramResourcePlan((_entry(1, "root/a"), _entry(2, "root/b")))
    finally:
        plan_mod._identity_digest = original_identity_digest

    original_max = plan_mod._UINT32_MAX
    plan_mod._UINT32_MAX = 1
    try:
        with pytest.raises(OverflowError, match="UINT32_MAX"):
            ProgramResourcePlan((_entry(1, "root/a"), _entry(2, "root/b")))
    finally:
        plan_mod._UINT32_MAX = original_max


def test_runtime_sized_bytes_are_explicitly_unresolved_until_host_materialization() -> None:
    plan = lower_program_resource_plan(
        _Program(_Value(1, attrs={"components": 1, "bytes": None, "runtime_sized": True}))
    )
    assert plan.maximum_bytes is None
    assert len(plan.entries) == 1
    row = plan.entries[0]
    assert row.runtime_sized is True
    assert row.resource_type == "runtime_sized"
    assert row.bytes is None
    assert row.maximum_bytes is None
    assert row.cells is None and row.itemsize is None
    before = plan.to_data()
    with pytest.raises(RuntimeError, match="materialized Program resource plan"):
        program_persistent_value_checkpoint_capacity(plan)
    assert plan.to_data() == before

    with pytest.raises(ValueError, match="unknown"):
        lower_program_resource_plan(
            SimpleNamespace(_values=(_Value(1),), _persistent_memory_limit=None)
        )
    with pytest.raises(ValueError, match="memory bound"):
        ProgramResourcePlan((_entry(1),), maximum_bytes="unknown")


def test_checkpoint_capacity_uses_native_materialization_for_runtime_sized_plan() -> None:
    key = ProgramPersistentValueKey(1, "root/0", "owner", "space", "clock")
    symbolic = ProgramResourcePlan(
        (ProgramResourcePlanEntry(key=key, bytes=None, runtime_sized=True),)
    )
    materialized_row = {
        "slot": 0,
        "key": {
            "value_id": 1,
            "occurrence_path_id": key.occurrence_path_id,
            "owner": 1,
            "space": 2,
            "clock": 3,
            "level": None,
        },
        "identity": "program-resource:v1:materialized-slot-0",
        "occurrence_path": "root/0",
        "owner_identity": "owner",
        "space_identity": "space",
        "clock_identity": "clock",
        "lifetime": 2,
        "centering": 1,
        "off_policy": 1,
        "spatial_transfer": 1,
        "components": 1,
        "ghosts": 0,
        "bytes": 8,
        "maximum_bytes": 32,
        "communicates": False,
        "restart_required": False,
        "communication": "none",
        "transfer_identity": "none",
        "restart_identity": "none",
        "component_names": "[\"value\"]",
        "shape": "[4]",
        "cells": 4,
        "itemsize": 8,
    }
    image = ProgramPersistentValueCheckpoint(
        bound=True,
        schema="program-persistent-value-checkpoint:v1",
        plan_schema="program-resource-plan:v1",
        plan_digest="b" * 64,
        maximum_bytes=32,
        slot_count=1,
        rows=(materialized_row,),
        metadata=(
            {
                "accepted_coordinate": 0,
                "cursor": 0,
                "accumulated_dt": 0.0,
                "topology_epoch": 0,
                "layout_generation": 0,
                "valid": False,
                "cold": True,
            },
        ),
        offsets=(0, 32),
        value_bytes=(0,),
        storage=b"\0" * 32,
    )
    payload = encode_program_persistent_value_checkpoint(image)

    class Native:
        def capture_program_persistent_value_checkpoint(self):
            return payload

    from pops.runtime._checkpoint_resource_budget import _persistent_checkpoint_capacity

    names, payload_capacity, evidence, materialized = _persistent_checkpoint_capacity(
        symbolic, target="amr_system", native=Native()
    )
    assert names
    assert payload_capacity > len(payload)
    assert evidence["encoded_capacity"] == len(payload)
    assert evidence["maximum_bytes"] == 32
    assert materialized.maximum_bytes == 32
    assert materialized.digest == "b" * 64

    before = symbolic.to_data()
    unbound = ProgramPersistentValueCheckpoint(
        bound=False,
        schema="program-persistent-value-checkpoint:v1",
        plan_schema="",
        plan_digest="",
        maximum_bytes=0,
        slot_count=0,
        rows=(),
        metadata=(),
        offsets=(),
        value_bytes=(),
        storage=b"",
    )
    unbound_payload = encode_program_persistent_value_checkpoint(unbound)

    class UnmaterializableNative:
        def capture_program_persistent_value_checkpoint(self):
            return unbound_payload

    with pytest.raises(RuntimeError, match="materialized native Program resource plan"):
        _persistent_checkpoint_capacity(
            symbolic, target="amr_system", native=UnmaterializableNative()
        )
    assert symbolic.to_data() == before


def test_resource_plan_has_one_canonical_surface_and_strict_serialization() -> None:
    key = ProgramPersistentValueKey(9, "root/0", "owner", "space", "clock")
    with pytest.raises(TypeError, match="unexpected keyword argument 'sizing'"):
        ProgramResourcePlanEntry(key=key, bytes=8, sizing="exact")
    with pytest.raises(ValueError, match="unsupported"):
        ProgramResourcePlanEntry(key=key, bytes=8, resource_type="static")
    with pytest.raises(ValueError, match="disagree"):
        ProgramResourcePlanEntry(
            key=key,
            bytes=None,
            runtime_sized=True,
            resource_type="exact",
        )
    with pytest.raises(TypeError, match="sealed ProgramResourcePlan"):
        plan_mod.persistent_slot(None, 9)

    for alias in (
        "PersistentValueKey",
        "ProgramResourceKey",
        "ProgramPersistentPlan",
        "ProgramPersistentPlanEntry",
        "build_program_resource_plan",
        "lower_program_persistent_plan",
        "program_persistent_plan",
        "program_resource_plan",
        "lower_resource_plan",
        "seal_program_resource_plan",
    ):
        assert not hasattr(plan_mod, alias)

    data = ProgramResourcePlan((_entry(9),)).to_data()
    del data["entries"][0]["resource_type"]
    with pytest.raises(ValueError, match="entry schema mismatch"):
        ProgramResourcePlan.from_data(data)

    data = ProgramResourcePlan((_entry(9),)).to_data()
    del data["schema_version"]
    with pytest.raises(ValueError, match="schema fields"):
        ProgramResourcePlan.from_data(data)


def test_emitted_resource_calls_use_slots_and_reject_legacy_ids_or_placeholders() -> None:
    plan = ProgramResourcePlan((_entry(42), _entry(7, "root/1")))
    source = "ctx.rhs_scratch(1, 0, u); ctx.cache_store_scratch(1, u);"
    _validate_resource_slot_calls(source, plan)
    assert "ctx.rhs_scratch(42," not in source
    with pytest.raises(ValueError, match="legacy"):
        _validate_resource_slot_calls("ctx.rhs_scratch(42, 0, u);", plan)
    with pytest.raises(ValueError, match="placeholder"):
        _validate_resource_slot_calls(
            "ctx.scratch_state(__POPS_PERSISTENT_SLOT_0__, 0, u);", plan
        )


def test_generated_field_route_is_dense_slot_prepared_and_deduplicated() -> None:
    plan = ProgramResourcePlan((_entry(42), _entry(7, "root/1")))
    prelude: list[str] = []
    var: dict[object, object] = {}
    _append_generated_field_route_preparation(
        prelude, var, slot="1", field="field/qualified", program_blocks=(0, 2)
    )
    _append_generated_field_route_preparation(
        prelude, var, slot="1", field="field/qualified", program_blocks=(0, 2)
    )
    assert prelude == [
        'ctx.prepare_generated_field_route(1, "field/qualified", {0, 2});'
    ]
    _validate_resource_slot_calls(
        'ctx.solve_fields_from_blocks_at(field_boundary_point_1, 1, "field/qualified", {});',
        plan,
    )
    with pytest.raises(ValueError, match="conflicting"):
        _append_generated_field_route_preparation(
            prelude, var, slot="1", field="field/other", program_blocks=(0, 2)
        )
    with pytest.raises(ValueError, match="legacy"):
        _validate_resource_slot_calls(
            'ctx.solve_fields_from_blocks_at(field_boundary_point_1, 42, "field/qualified", {});',
            plan,
        )


def test_symbolic_candidate_serializes_only_the_abi_unknown_sentinel() -> None:
    class _CandidateGraph:
        name = "runtime"
        _histories = {}
        _history_persistence = {}
        _history_blocks = {}
        _history_spaces = {}

        @staticmethod
        def _block_indices() -> dict[Any, int]:
            return {}

    key = ProgramPersistentValueKey(42, "root/0", "owner", "space", "clock")
    plan = ProgramResourcePlan(
        (ProgramResourcePlanEntry(key=key, bytes=None, runtime_sized=True),)
    )
    tables = _emit_candidate_tables(
        _CandidateGraph(), None, (), "routes", "system", plan=plan
    )
    assert "ProgramResourcePlanType::runtime_sized" in tables
    assert "kProgramResourcePlanUnknownExtent" in tables
    assert "sizeof(ProgramCandidateState)" not in tables
    install = _emit_system_install(
        "system", "", "", "", artifact_identity="a", route_manifest="r",
        program_name="runtime", maximum_bytes=None,
    )
    assert (
        "descriptor.maximum_bytes = "
        "pops::runtime::program::kProgramResourcePlanUnknownExtent;" in install
    )
    assert "sizeof(ProgramCandidateState)" not in install
    amr = _emit_amr_candidate_entry_suffix(None)
    assert "descriptor.maximum_bytes = pops::runtime::program::kProgramResourcePlanUnknownExtent;" in amr
    assert "sizeof(ProgramCandidateState)" not in amr


def test_empty_value_plan_keeps_its_zero_manifest_but_uses_a_symbolic_descriptor() -> None:
    plan = ProgramResourcePlan(())
    assert plan.maximum_bytes == 0
    install = _emit_system_install(
        "system", "", "", "", artifact_identity="a", route_manifest="r",
        program_name="empty", maximum_bytes=None,
    )
    assert "descriptor.maximum_bytes = pops::runtime::program::kProgramResourcePlanUnknownExtent;" in install


def test_amr_level_schedule_requires_transfer_provider() -> None:
    from pops.time import AMRLevel, Clock, Every, Schedule, TimePoint, Zero

    clock = Clock("macro")
    schedule = Schedule(Every(AMRLevel(clock, level=1), 2), off=Zero())
    value = _Value(
        4,
        clock=clock,
        point=TimePoint(clock),
        attrs={"components": 1, "bytes": 8, "schedule": schedule},
    )
    with pytest.raises(ValueError, match="transfer provider"):
        lower_program_resource_plan(_Program(value), target="amr_system")


def test_accumulate_dt_uses_its_canonical_manifest_tag() -> None:
    from pops.time import AcceptedStep, AccumulateDt, Clock, Every, Schedule, TimePoint

    clock = Clock("macro")
    schedule = Schedule(Every(AcceptedStep(clock), 2), off=AccumulateDt())
    value = _Value(
        4,
        clock=clock,
        point=TimePoint(clock),
        attrs={"components": 1, "bytes": 8, "schedule": schedule},
    )
    plan = lower_program_resource_plan(_Program(value))
    assert plan.entries[0].off_policy == "accumulate_dt"


def test_resource_rows_expand_dense_by_occurrence_and_resolved_level() -> None:
    value = _Value(
        4,
        attrs={
            "components": 1,
            "resolved_levels": (0, 1),
            "bytes": (8, 16),
            "maximum_bytes": (12, 24),
            "cells": (1, 2),
            "itemsize": 8,
            "shape_by_level": ((1,), (2,)),
        },
    )
    plan = lower_program_resource_plan(_Program(value))
    assert [row.key.level for row in plan] == [0, 1]
    assert [row.slot for row in plan] == [0, 1]
    assert [row.bytes for row in plan] == [8, 16]
    assert [row.maximum_bytes for row in plan] == [12, 24]
    assert plan.slot_for_value(value, level=0) == 0
    assert plan.slot_for_value(value, level=1) == 1
    with pytest.raises(KeyError, match="resolved level"):
        plan.slot_for_value(value)


def test_resolved_layout_derives_an_exact_ceiling_without_a_byte_hint() -> None:
    value = _Value(
        8,
        attrs={
            "components": 3,
            "shape": (4, 5),
            "itemsize": 8,
        },
    )

    plan = lower_program_resource_plan(_Program(value))

    assert plan.maximum_bytes == 3 * 4 * 5 * 8
    row = plan.entries[0]
    assert row.runtime_sized is False
    assert row.shape == (4, 5)
    assert row.cells == 20
    assert row.itemsize == 8
    assert row.bytes == row.maximum_bytes == plan.maximum_bytes


def test_ghosted_geometry_requires_an_allocation_shape_for_an_exact_ceiling() -> None:
    symbolic = _Value(
        8,
        attrs={
            "components": 1,
            "shape": (4, 5),
            "itemsize": 8,
            "ghosts": 1,
        },
    )
    exact = _Value(
        9,
        attrs={
            "components": 1,
            "shape": (4, 5),
            "allocation_shape": (6, 7),
            "itemsize": 8,
            "ghosts": 1,
        },
    )

    symbolic_row = lower_program_resource_plan(_Program(symbolic)).entries[0]
    exact_plan = lower_program_resource_plan(_Program(exact))
    exact_row = exact_plan.entries[0]

    assert symbolic_row.runtime_sized is True
    assert symbolic_row.bytes is None
    assert exact_row.runtime_sized is False
    assert exact_row.shape == (4, 5)
    assert exact_row.cells == 42
    assert exact_row.bytes == exact_row.maximum_bytes == exact_plan.maximum_bytes == 42 * 8


def test_source_byte_hint_without_complete_layout_remains_host_materialized() -> None:
    value = _Value(8, attrs={"components": 1, "bytes": 64})

    row = lower_program_resource_plan(_Program(value)).entries[0]

    assert row.runtime_sized is True
    assert row.bytes is None
    assert row.maximum_bytes is None


def test_explicit_exact_layout_without_geometry_is_refused_before_emission() -> None:
    value = _Value(
        8,
        attrs={"components": 1, "bytes": 64, "resource_type": "exact"},
    )

    with pytest.raises(ValueError, match="exact layout is incomplete"):
        lower_program_resource_plan(_Program(value))


def test_field_solve_alias_is_omitted_without_a_materialized_slot() -> None:
    value = _Value(
        4,
        op="solve_fields",
        attrs={
            "components": 1,
            "resolved_levels": (0,),
        },
    )
    plan = lower_program_resource_plan(_Program(value))
    # ``solve_fields(state)`` refreshes the provider-owned auxiliary field and
    # then aliases its input state.  It emits neither ``prepare_*`` nor a
    # dense route token, so retaining it as a runtime-sized resource would
    # leave host materialization with an intentionally unprimed slot.
    assert plan.entries == ()


def test_reachability_excludes_dead_matrix_free_operator_but_retains_indirect_consumer() -> None:
    dead_apply = _Value(12, op="apply", attrs={"components": 1, "bytes": 8}, name="dead_apply")
    dead_operator = _Value(
        11,
        op="coupled_interface_jacobian",
        attrs={"apply_block": (dead_apply,)},
        name="dead_operator",
    )
    live_apply = _Value(22, op="apply", attrs={"components": 1, "bytes": 16}, name="live_apply")
    live_operator = _Value(
        21,
        op="coupled_interface_jacobian",
        attrs={"apply_block": (live_apply,)},
        name="live_operator",
    )
    solve = _Value(
        30,
        op="solve_coupled_implicit",
        attrs={"components": 1, "bytes": 32},
        inputs=(live_operator,),
        name="solve",
    )

    class ExecutableProgram(_Program):
        def __init__(self) -> None:
            super().__init__(dead_operator, live_operator, solve)
            self._commits = (solve,)

    plan = lower_program_resource_plan(ExecutableProgram())
    paths = {row.key.occurrence_path for row in plan}
    assert "root/0" not in paths
    assert "root/0/apply_block/0" not in paths
    assert "root/1" in paths
    assert "root/1/apply_block/0" in paths
    assert "root/2" in paths


def test_reachability_retains_values_referenced_through_executable_control_regions() -> None:
    indirect = _Value(41, attrs={"components": 1, "bytes": 24}, name="indirect")
    control = _Value(
        42,
        op="branch",
        attrs={"true_block": (indirect,), "components": 1, "bytes": 8},
        name="control",
    )

    class ExecutableProgram(_Program):
        def __init__(self) -> None:
            super().__init__(control)
            self._commits = (control,)

    plan = lower_program_resource_plan(ExecutableProgram())
    assert {row.key.occurrence_path for row in plan} == {
        "root/0",
        "root/0/true_block/0",
    }


def test_at_end_and_missing_off_policy_are_refused_before_emission() -> None:
    from pops.time import AcceptedStep, AtEnd, Clock, Every, Schedule, TimePoint

    clock = Clock("macro")
    at_end = Schedule(AtEnd(AcceptedStep(clock)))
    value = _Value(
        4,
        clock=clock,
        point=TimePoint(clock),
        attrs={"components": 1, "bytes": 8, "schedule": at_end},
    )
    with pytest.raises(NotImplementedError, match="AtEnd"):
        lower_program_resource_plan(_Program(value))

    every = Schedule(Every(AcceptedStep(clock), 2))
    value = _Value(
        5,
        clock=clock,
        point=TimePoint(clock),
        attrs={"components": 1, "bytes": 8, "schedule": every},
    )
    with pytest.raises(ValueError, match="OffPolicy"):
        lower_program_resource_plan(_Program(value))
