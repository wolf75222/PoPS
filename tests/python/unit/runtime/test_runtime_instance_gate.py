"""ADC-687: one installed runtime and accepted-only exact consumers."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pops
import pytest

from pops.codegen._plans import BindInputs, InstallPlan
from pops.codegen._compiled_artifact import (
    CompiledLayoutProgram,
    CompiledSimulationArtifact,
)
from pops.identity import Identity, make_identity
from pops.model import Handle, OwnerPath
from pops.output import (
    AsyncScientificOutput,
    LiveVisualization,
    NPZ,
    NPZWriter,
    OutputPublicationReceipt,
    RaiseOnFlush,
    ReportOnly,
    read_npz,
)
from pops.output._consumer_contracts import (
    ConsumerCursorSet,
    ConsumerGraph,
    ConsumerKind,
    ConsumerManifest,
    ConsumerQuantity,
    ParallelMode,
)
from pops.output._restart_provider import ReopenedRestart, RestartV3
from pops.output.observers import ObserverReceipt
from pops.runtime._runtime_instance import RuntimeInstance
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import (
    AcceptedStep,
    AdaptiveCFL,
    Always,
    AtEnd,
    AtStart,
    Clock,
    Every,
    ExternalTimeGrid,
    FixedDt,
    Schedule,
    When,
    every_dt,
)
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.unit.runtime.test_runtime_planning import _artifact as _planning_artifact


def _path_owner(path: Path) -> tuple[int, int]:
    status = path.lstat()
    return int(status.st_dev), int(status.st_ino)


def _install(names=("fluid",), *, heterogeneous=False, memory_spaces=("host",)):
    """Build the planning fixture against the exact loaded native ABI and resources."""
    from pops import _pops

    template = _planning_artifact(names, heterogeneous=heterogeneous, memory_spaces=memory_spaces)
    native_abi = _pops.abi_key()
    if not isinstance(native_abi, str) or not native_abi:
        raise RuntimeError("loaded native runtime exposes no authenticated ABI key")
    for block in template.blocks:
        block.model.abi_key = native_abi
    for row in template.layout_programs:
        row.program.abi_key = native_abi
    layout_programs = tuple(
        CompiledLayoutProgram(row.layout_id, row.target, row.block_names, row.program)
        for row in template.layout_programs
    )
    program = layout_programs[0].program if len(layout_programs) == 1 else None
    artifact = CompiledSimulationArtifact(
        template.plan,
        program,
        template.blocks,
        layout_programs,
        template.component_artifacts,
    )
    inputs = BindInputs()
    return InstallPlan(
        artifact=artifact,
        bind_inputs=inputs,
        instances={
            block.name: {"model": block.model, "spatial": block.spatial}
            for block in artifact.blocks
        },
        params=artifact.bind_schema.resolve_bind({}, compile_values=artifact.plan.compile_values),
        aux={},
        execution_context=artifact_execution_context(artifact),
    )


class _Executor:
    def __init__(self, plan: InstallPlan) -> None:
        self._plan = plan
        self._s = self
        geometry = plan.artifact.layout_plan.layouts[0].geometry
        self._nx, self._ny = geometry.cells
        self._time = 0.0
        self._step = 0
        self._last_run_manifest = None
        self._last_run_identity = None
        self._restart_lineage_identity = None
        self._last_restart_identity = None
        self._prepared_restart_bit_identical = None
        self._step_snapshot = None
        self._step_committed = False
        self._step_strategy = FixedDt(1.0)
        self._step_controller = None
        self._step_transaction_plan = None
        self._last_step_transaction_report = None
        self.bound_snapshot = SimpleNamespace(
            semantic_identity=plan.artifact.semantic_identity,
            artifact_identity=plan.artifact.artifact_identity,
            bind_identity=plan.bind_identity,
        )
        from pops.runtime._checkpoint_resource_budget import (
            _producer_checkpoint_resource_budget,
        )

        self._checkpoint_resource_budget = _producer_checkpoint_resource_budget(
            {
                "t": np.asarray(0.0),
                "macro_step": np.asarray(0, dtype=np.int64),
                "abi_key": np.asarray("x" * 512),
                "runtime_consumer_graph": np.asarray("x" * 512),
                "runtime_consumer_cursors": np.asarray("x" * 4096),
                "runtime_consumer_diagnostics": np.asarray("x" * 4096),
                "pops_checkpoint_manifest": np.asarray("x" * 32768),
                "pops_restart_identity": np.asarray("x" * 128),
                "temporal_restart_state": np.asarray("x" * 32768),
            },
            runtime_kind="uniform",
            authority=plan.bind_identity.token,
        )
        graph = plan.artifact.plan.consumer_graph
        clocks = (
            sorted(
                {node.schedule.domain.clock for node in graph.nodes},
                key=lambda clock: clock.qualified_id,
            )
            if graph is not None
            else []
        )
        self._temporal_restart_state = TemporalRestartState()
        if clocks:
            if len(clocks) != 1:
                raise ValueError("test executor requires one authored consumer clock")
            clock = clocks[0]
            self._temporal_restart_state.configure_program(
                {
                    "schema_version": 1,
                    "kind": "pops.temporal-program-schedule",
                    "primary_clock": clock.qualified_id,
                    "clocks": [
                        {
                            "id": clock.qualified_id,
                            "descriptor": clock.to_data(),
                            "ticks_per_macro": 1,
                        }
                    ],
                    "subcycles": [],
                    "synchronizations": [],
                    "schedules": [],
                    "histories": [],
                },
                time=0.0,
                macro_step=0,
            )
        from pops.runtime._step_strategy import prepare_program_run

        prepared_run = prepare_program_run(self)
        self._temporal_restart_state.begin_run(
            prepared_run.restart_payload,
            time=self._time,
            macro_step=self._step,
        )

    @property
    def last_run_identity(self):
        return self._last_run_identity

    @property
    def last_restart_identity(self):
        return self._last_restart_identity

    def _checkpoint_identities(self):
        return (
            self.bound_snapshot.semantic_identity,
            self.bound_snapshot.artifact_identity,
            self.bound_snapshot.bind_identity,
        )

    def _restore_checkpoint_run_identity(self, identity):
        assert type(identity) is Identity
        assert identity.domain == "run"
        self._last_run_identity = Identity.from_data(identity.to_data())

    def time(self):
        return self._time

    def macro_step(self):
        return self._step

    def step(self, dt):
        self._time += float(dt)
        self._step += 1

    def _begin_step_transaction(self):
        self._step_snapshot = (self._time, self._step)
        self._step_committed = False

    def _commit_step_transaction(self):
        if self._step_snapshot is None:
            raise RuntimeError("missing transaction")
        self._step_committed = True

    def _finalize_step_transaction(self):
        if self._step_snapshot is None or not self._step_committed:
            raise RuntimeError("missing committed transaction")
        self._step_snapshot = None
        self._step_committed = False

    def _rollback_step_transaction(self):
        self._time, self._step = self._step_snapshot
        self._step_snapshot = None
        self._step_committed = False

    def nx(self):
        return self._nx

    def ny(self):
        return self._ny

    def block_names(self):
        return ["fluid"]

    def variable_names(self, block, space):
        assert block == "fluid" and space == "conservative"
        return ["rho"]

    def state_global(self, block):
        assert block == "fluid"
        return np.full(self._nx * self._ny, self._step + 1.0)

    def local_boxes(self, block):
        assert block == "fluid"
        return [((0, 0), (self._nx, self._ny))]

    def spatial_shape(self):
        return self._nx, self._ny

    def _output_geometry_snapshot(self, origin, spacing, shape, cell_measure):
        assert tuple(shape) == (self._nx, self._ny)
        assert cell_measure == "pops://cell-measures/cartesian-area@1"
        cell_shape = tuple(reversed(shape))
        valid = np.ones(cell_shape, dtype=np.bool_)
        coverage = np.zeros(cell_shape, dtype=np.bool_)
        volumes = np.full(cell_shape, spacing[0] * spacing[1], dtype=np.float64)
        for value in (valid, coverage, volumes):
            value.setflags(write=False)
        return {
            "dimension": len(shape),
            "topology_epoch": 0,
            "cell_shape": cell_shape,
            "boxes": ((0, 0, cell_shape[0], cell_shape[1]),),
            "valid_cells": valid,
            "coverage": coverage,
            "cell_volumes": volumes,
        }

    def local_state(self, block, index):
        assert block == "fluid" and index == 0
        return self.state_global(block).reshape(1, self._ny, self._nx)

    def output_state_local_pieces(self, block, level):
        assert block == "fluid" and level == 0
        return (
            {
                "lower": (0, 0),
                "upper": (self._ny, self._nx),
                "values": np.ascontiguousarray(
                    self.state_global(block).reshape(1, self._ny, self._nx),
                    dtype=np.float64,
                ),
                "global_box_index": 0,
                "owner_rank": 0,
                "replicated": False,
            },
        )

    def output_state_root_pieces(self, communicator, block, level):
        """Expose the exact duplicated consumer lane required by ROOT publication tests."""
        from pops._native_collectives import require_communicator, size

        expected = self._plan.execution_context.communicator
        native = require_communicator(communicator, allow_world=False)
        if expected.identity != "MPI_COMM_WORLD":
            raise ValueError("ROOT gather requires an MPI execution context")
        if size(native) != 1:
            raise RuntimeError(
                "runtime-instance unit executor only implements a singleton ROOT gather"
            )
        return self.output_state_local_pieces(block, level)

    def reduce_component(self, block, kind, component):
        assert (block, kind, component) == ("fluid", "sum", 0)
        return float(np.sum(self.state_global(block)))

    def checkpoint(self, path):
        from pops.runtime._checkpoint_manifest import seal_checkpoint_payload
        from pops.runtime._engine_descriptors import abi_key

        payload = {
            "t": self._time,
            "macro_step": self._step,
            "abi_key": abi_key(),
            "temporal_restart_state": self._temporal_restart_state.checkpoint_json(
                time=self._time,
                macro_step=self._step,
            ),
        }
        seal_checkpoint_payload(self, payload, runtime_kind="uniform")
        target = path if str(path).endswith(".npz") else str(path) + ".npz"
        with open(target, "wb") as stream:
            np.savez_compressed(stream, **payload)
        return target

    def restart(self, path):
        from pops.output._checkpoint_collective import _bounded_checkpoint_path_bytes
        from pops.runtime._checkpoint_resource_budget import require_checkpoint_resource_budget

        budget = require_checkpoint_resource_budget(self).max_archive_bytes
        prepared = self._prepare_checkpoint_restart(
            _bounded_checkpoint_path_bytes(Path(path), budget), bit_identical=False
        )
        self._begin_checkpoint_restart()
        result = self._apply_checkpoint_restart(prepared)
        self._commit_checkpoint_restart()
        self._finalize_checkpoint_restart()
        return result

    def _prepare_checkpoint_restart(self, payload, *, bit_identical):
        from pops.output._checkpoint_collective import decode_checkpoint_bytes
        from pops.runtime._checkpoint_resource_budget import require_checkpoint_resource_budget
        from pops.runtime._checkpoint_manifest import authenticate_checkpoint_payload

        if type(bit_identical) is not bool:
            raise TypeError("test restart bit_identical must be an exact bool")
        self._prepared_restart_bit_identical = bit_identical
        stored = decode_checkpoint_bytes(payload, require_checkpoint_resource_budget(self))
        identity = authenticate_checkpoint_payload(self, stored, runtime_kind="uniform")
        temporal = TemporalRestartState.from_json(
            stored["temporal_restart_state"],
            time=stored["t"],
            macro_step=stored["macro_step"],
            program_schedule=self._temporal_restart_state.program_schedule,
        )
        return identity, float(stored["t"]), int(stored["macro_step"]), temporal

    def _begin_checkpoint_restart(self):
        self._begin_step_transaction()
        self._restart_identity_snapshot = self._last_restart_identity
        self._restart_temporal_snapshot = self._temporal_restart_state

    def _apply_checkpoint_restart(self, prepared):
        identity, self._time, self._step, self._temporal_restart_state = prepared
        self._last_restart_identity = identity
        return identity

    def _commit_checkpoint_restart(self):
        self._commit_step_transaction()

    def _finalize_checkpoint_restart(self):
        self._finalize_step_transaction()
        del self._restart_identity_snapshot
        del self._restart_temporal_snapshot

    def _rollback_checkpoint_restart(self):
        self._rollback_step_transaction()
        self._last_restart_identity = self._restart_identity_snapshot
        self._temporal_restart_state = self._restart_temporal_snapshot
        del self._restart_identity_snapshot
        del self._restart_temporal_snapshot


class _CustomNPZ:
    __pops_ir_immutable__ = True

    def __init__(self, mode: ParallelMode) -> None:
        if type(mode) is not ParallelMode:
            raise TypeError("custom NPZ test provider requires an exact ParallelMode")
        self._mode = mode

    def consumer_data(self):
        return {
            "schema_version": 1,
            "provider_id": "pops.test.custom-npz.v1",
            "format_name": "npz",
            "extension": ".npz",
            "parallel_mode": self._mode.value,
        }

    def writer(self):
        return NPZWriter(self._mode)


def _scientific_output_mode(artifact: CompiledSimulationArtifact) -> ParallelMode:
    """Select an explicit publication mode compatible with the sealed native artifact."""
    communicator = artifact.platform_manifest.communicator.require(
        "runtime-instance fixture communicator"
    )
    if communicator == "serial":
        return ParallelMode.SERIAL
    if communicator == "MPI_COMM_WORLD":
        return ParallelMode.ROOT
    raise ValueError("unsupported runtime-instance fixture communicator %r" % communicator)


def _with_graph(
    tmp_path,
    *,
    kind=ConsumerKind.SCIENTIFIC_OUTPUT,
    output_format=None,
    target_uri=None,
    operation=None,
    schedule=None,
):
    base = _install()
    parallel_mode = _scientific_output_mode(base.artifact)
    if isinstance(output_format, type):
        output_format = output_format(parallel_mode)
    layout = base.artifact.layout_plan.layouts[0].handle
    clock = Clock("solution", owner=OwnerPath.consumer("adc-687"))
    quantity = ConsumerQuantity(
        Handle("rho", kind="state", owner=OwnerPath.model("adc-687")),
        "state:u",
        layout.qualified_id,
    )
    resolved_mode = (
        parallel_mode
        if kind is ConsumerKind.SCIENTIFIC_OUTPUT
        else (
            ParallelMode(operation.consumer_data()["parallel_mode"])
            if kind is ConsumerKind.MONITOR
            else ParallelMode.SERIAL
        )
    )
    manifest = ConsumerManifest(
        Handle("density", kind="consumer", owner=OwnerPath.consumer("adc-687")),
        kind,
        (quantity,),
        Schedule(Every(AcceptedStep(clock), 1)) if schedule is None else schedule(clock),
        str(tmp_path) if target_uri is None else str(target_uri),
        NPZ(mode=parallel_mode)
        if output_format is None and kind is ConsumerKind.SCIENTIFIC_OUTPUT
        else output_format,
        resolved_mode,
        operation=operation,
    )
    graph = ConsumerGraph((manifest,))
    from pops.output._restart_provider import RestartAuthority

    record = replace(
        base.artifact.plan,
        consumer_graph=graph,
        restart_authority=RestartAuthority.from_consumer_graph(graph),
    )
    artifact = CompiledSimulationArtifact(record, base.artifact.program, base.artifact.blocks)
    inputs = BindInputs()
    plan = InstallPlan(
        artifact=artifact,
        bind_inputs=inputs,
        instances={
            block.name: {"model": block.model, "spatial": block.spatial}
            for block in artifact.blocks
        },
        params=artifact.bind_schema.resolve_bind({}, compile_values=artifact.plan.compile_values),
        aux={},
        execution_context=artifact_execution_context(artifact),
    )
    return plan, graph, manifest


def test_runtime_instance_retains_complete_multilayout_plan_without_target_dispatch():
    plan = _install(("fluid", "solid"), heterogeneous=True)
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    assert runtime._layout_plan is plan.artifact.layout_plan
    assert runtime._runtime_plan.layout_plan_id == runtime._layout_plan.qualified_id
    assert len(runtime._runtime_plan.calls) == 2
    assert len(runtime._runtime_plan.communication.transfers) == 1
    assert (
        runtime._runtime_plan.communication.transfers[0].provider_id
        == runtime._layout_plan.mappings[0].provider_id
    )


def test_runtime_instance_refuses_executor_without_checkpoint_resource_authority():
    plan = _install(("fluid", "solid"), heterogeneous=True)

    with pytest.raises(TypeError, match="lacks its authenticated checkpoint resource authority"):
        RuntimeInstance(plan, executor=object())


def test_checkpoint_budget_uses_authenticated_artifact_block_metadata():
    """The local Program is not a substitute for the artifact's block authority."""
    from pops.runtime._checkpoint_resource_budget import _program_for_install
    from tests.python.unit.codegen._typed_artifact_fixture import CompiledComponent

    template = _planning_artifact(("ions", "electrons"))
    local_program = CompiledComponent("checkpoint-budget-local-program", target="system")
    local_program.program = object()
    local_program.program_block_routes = ((0, "ions"), (1, "electrons"))

    def reject_local_arguments():
        raise AssertionError("checkpoint budgeting must not read local Program arguments")

    local_program.arguments = reject_local_arguments
    layout_program = CompiledLayoutProgram(
        template.layout_programs[0].layout_id,
        "system",
        ("ions", "electrons"),
        local_program,
    )
    artifact = CompiledSimulationArtifact(
        template.plan,
        local_program,
        template.blocks,
        (layout_program,),
    )
    artifact.verify()

    program, components = _program_for_install(SimpleNamespace(artifact=artifact))

    assert program is local_program.program
    assert components == {"ions": 1, "electrons": 1}


def test_uniform_checkpoint_budget_reserves_lazy_schedule_cache_from_program_authority():
    from pops.runtime._checkpoint_resource_budget import _common_budget

    class Native:
        def __init__(self, live_nodes=(), name="node_17"):
            self.live_nodes = live_nodes
            self.name = name

        @staticmethod
        def variable_names(block, space):
            assert (block, space) == ("fluid", "conservative")
            return ("rho",)

        def program_cache_nodes(self):
            return self.live_nodes

        def program_cache_name(self, node):
            assert node == 17
            return self.name

        @staticmethod
        def program_cache_ncomp(node):
            assert node == 17
            return 1

        @staticmethod
        def program_cache_ngrow(node):
            assert node == 17
            return 9

    class Program:
        def __init__(self, cache_required):
            self.cache_required = cache_required
            self._histories = {}
            self._histories_ncomp = {}
            self._history_blocks = {}

        def temporal_manifest(self):
            return {
                "schedules": [
                    {"node_id": 17, "schedule": {"kind": "hold"}, "cache_required": self.cache_required}
                ]
            }

    install_plan = SimpleNamespace(
        artifact=SimpleNamespace(
            artifact_identity=SimpleNamespace(token="artifact"),
            plan=SimpleNamespace(consumer_graph=None),
        ),
        bind_identity=SimpleNamespace(token="bind"),
    )

    def budget(owner, program):
        return _common_budget(
            owner,
            install_plan,
            runtime_kind="uniform",
            cells=(6,),
            shape=(2, 3),
            rank_capacity=1,
            auxiliary_metadata_bytes=0,
            auxiliary_components=0,
            accepted_program_bytes=0,
            source_authority_bytes=0,
            structural_bytes=0,
            field_provider_manifest_characters=0,
            program=program,
            block_nvars_by_name={"fluid": 1},
            field_names=(),
        )

    no_cache = budget(SimpleNamespace(_s=Native()), Program(cache_required=False))
    before_run = budget(SimpleNamespace(_s=Native()), Program(cache_required=True))
    after_run = budget(SimpleNamespace(_s=Native((17,))), Program(cache_required=True))

    assert before_run.max_members == no_cache.max_members + 5
    assert before_run == after_run
    assert before_run.max_uncompressed_bytes > no_cache.max_uncompressed_bytes
    with pytest.raises(ValueError, match="noncanonical exact node name"):
        budget(SimpleNamespace(_s=Native((17,), name="held-density")), Program(cache_required=True))


def test_checkpoint_history_budget_uses_installed_storage_depth_not_program_lookback():
    from pops.runtime._checkpoint_resource_budget import _history_capacity

    class Program:
        _histories = {"blk.rate": 1}
        _histories_ncomp = {"blk.rate": 1}
        _history_blocks = {}

    class Native:
        @staticmethod
        def history_names():
            return ["blk.rate"]

        @staticmethod
        def history_depth(name):
            assert name == "blk.rate"
            return 2

        @staticmethod
        def history_ncomp(name):
            assert name == "blk.rate"
            return 1

    names, _bytes, evidence = _history_capacity(
        Program(), cells=(8, 32), amr=True, block_nvars={"blk": 1}, native=Native()
    )
    assert evidence == (("blk.rate", 1, 2),)
    assert "history_blk.rate_level_0_0" in names
    assert "history_blk.rate_level_0_1" in names
    assert "history_blk.rate_level_1_0" in names
    assert "history_blk.rate_level_1_1" in names

    class WrongNames(Native):
        @staticmethod
        def history_names():
            return ["other.rate"]

    with pytest.raises(ValueError, match="names differ"):
        _history_capacity(
            Program(), cells=(8,), amr=False, block_nvars={"blk": 1}, native=WrongNames()
        )

    class WrongComponents(Native):
        @staticmethod
        def history_ncomp(name):
            return 2

    with pytest.raises(ValueError, match="component width differs"):
        _history_capacity(
            Program(), cells=(8,), amr=False, block_nvars={"blk": 1}, native=WrongComponents()
        )


def test_checkpoint_budget_projects_opaque_consumer_evidence_to_canonical_hex():
    from pops.identity import canonical_bytes
    from pops.runtime._checkpoint_resource_budget import _consumer_evidence

    consumer_data = {
        "schema_version": 2,
        "nodes": [{"operation": {"opaque_state": b"\x00\xff\x80checkpoint"}}],
    }
    graph = SimpleNamespace(
        identity=SimpleNamespace(token="consumer-graph-identity"),
        nodes=(),
        to_data=lambda: consumer_data,
    )
    install_plan = SimpleNamespace(
        artifact=SimpleNamespace(plan=SimpleNamespace(consumer_graph=graph))
    )

    identity, count, evidence = _consumer_evidence(install_plan)

    assert (identity, count) == ("consumer-graph-identity", 0)
    assert evidence == {
        "encoding": "pops-canonical-cbor-hex-v1",
        "payload": canonical_bytes(consumer_data).hex(),
    }
    assert bytes.fromhex(evidence["payload"]) == canonical_bytes(consumer_data)
    json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)


def test_checkpoint_zip64_capacity_uses_reviewed_checked_formula():
    from pops.runtime._checkpoint_resource_budget import (
        _amr_field_provider_manifest_capacity,
        _archive_byte_capacity,
        _checkpoint_member_names,
    )

    uncompressed = 4_321_987
    names = ("state_fluid", "pops_checkpoint_manifest", "pops_restart_identity")

    assert _archive_byte_capacity(uncompressed, names, where="test") == (
        uncompressed
        + (uncompressed >> 12)
        + (uncompressed >> 14)
        + (uncompressed >> 25)
        + 145 * len(names)
        + 2 * sum(map(len, names))
        + 98
    )

    field_row = [
        "pops.amr.field-provider-checkpoint-manifest@1",
        "field-slot-\N{GREEK SMALL LETTER PHI}",
        "2",
        "provider-identity",
        "plan-identity",
        "configuration-identity",
        "17",
        "23",
        "materialized",
        "output-owner",
        "fluid",
        "phi",
        "1",
        "dependency-identity",
        "fluid",
        "rho",
        "1",
        "fluid",
        "boundary-phi",
    ]
    owner = SimpleNamespace(
        _s=SimpleNamespace(field_provider_checkpoint_manifest=lambda: [field_row])
    )
    field_slots, manifest_characters, structural_bytes = _amr_field_provider_manifest_capacity(
        owner, configured_levels=12
    )
    maximal_row = list(field_row)
    maximal_row[2] = "12"
    maximal_row[6] = str((1 << 64) - 1)
    maximal_row[7] = str((1 << 64) - 1)
    maximal_row[8] = "unmaterialized"
    maximal_text = json.dumps((tuple(maximal_row),), separators=(",", ":"), ensure_ascii=True)

    assert field_slots == (field_row[1],)
    assert manifest_characters == len(maximal_text)
    assert structural_bytes == len(maximal_text) * np.dtype("U1").itemsize
    uniform_member_names = _checkpoint_member_names(
        runtime_kind="uniform",
        block_names=("fluid",),
        field_names=(),
        history_names=(),
        cache_names=(),
        levels=1,
        rank_capacity=1,
    )
    member_names = _checkpoint_member_names(
        runtime_kind="amr",
        block_names=("fluid",),
        field_names=field_slots,
        history_names=(),
        cache_names=(),
        levels=12,
        rank_capacity=2,
    )
    assert uniform_member_names.count("checkpoint_migration") == 1
    assert member_names.count("checkpoint_migration") == 0
    assert member_names.count("field_provider_manifest") == 1


@pytest.mark.parametrize(
    ("shape", "bounds"),
    (
        ((7,), ((1,), (6,))),
        ((7, 9), ((1, 2), (6, 8))),
        ((7, 9, 11), ((1, 2, 3), (6, 8, 10))),
    ),
)
def test_local_boxes_preserve_exact_ranked_half_open_bounds(shape, bounds):
    runtime = object.__new__(RuntimeInstance)
    runtime._executor = SimpleNamespace(
        spatial_shape=lambda: shape,
        local_boxes=lambda block: (bounds,) if block == "fluid" else (),
    )

    assert runtime.local_boxes("fluid") == (bounds,)


def test_local_boxes_reject_a_fixed_rank_or_non_integral_provider_shape():
    runtime = object.__new__(RuntimeInstance)
    runtime._executor = SimpleNamespace(
        spatial_shape=lambda: (7, 9, 11),
        local_boxes=lambda _block: (((0, 0), (7, 9)),),
    )
    with pytest.raises(TypeError, match="exact rank 3"):
        runtime.local_boxes("fluid")

    runtime._executor = SimpleNamespace(
        spatial_shape=lambda: (7,),
        local_boxes=lambda _block: (((0.0,), (7,)),),
    )
    with pytest.raises(TypeError, match="plain integer"):
        runtime.local_boxes("fluid")


def test_private_engines_expose_no_scientific_output_policy_surface():
    import ast

    root = Path(__file__).resolve().parents[4]
    for relative in (
        "python/pops/runtime/_system_io.py",
        "python/pops/runtime/_amr_system_io.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        methods = {
            node.name
            for statement in tree.body
            if isinstance(statement, ast.ClassDef)
            for node in statement.body
            if isinstance(node, ast.FunctionDef)
        }
        assert "write" not in methods
        assert "_write_hdf5_parallel" not in methods


def test_runtime_instance_inspection_exposes_install_and_consumer_evidence():
    plan = _install()
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    report = runtime.inspect()
    payload = report.to_dict()
    assert payload["runtime"] == "uniform"
    assert payload["instance"]["bind_identity"] == plan.bind_identity.to_data()
    assert payload["instance"]["plan_identity"] == plan.artifact.plan.plan_identity.to_data()
    assert payload["instance"]["resolved_dimension"] == 2
    assert payload["instance"]["supported_dimensions"] == [2]
    assert payload["runtime_environment"]["dimension"] == 2
    assert payload["runtime_environment"]["supported_dimensions"] == [2]
    assert payload["instance"]["native_spatial_layouts"] == {
        layout_id: row.to_data() for layout_id, row in plan.artifact.native_layouts.items()
    }
    assert payload["instance"]["runtime_plan"] == runtime._runtime_plan.to_data()
    assert (
        payload["instance"]["runtime_plan"]["communication"]["layout_plan_id"]
        == plan.artifact.layout_plan.qualified_id
    )
    assert (
        payload["instance"]["runtime_plan"]["resources"]["execution_context_identity"]
        == plan.execution_context.identity.to_data()
    )
    assert (
        payload["instance"]["runtime_plan"]["determinism"]["execution_context_identity"]
        == plan.execution_context.identity.to_data()
    )
    assert payload["instance"]["consumer_graph"] == runtime.consumer_graph.to_data()
    assert (
        payload["instance"]["restart_authority"] == plan.artifact.plan.restart_authority.to_data()
    )
    assert runtime._restart_operation() is plan.artifact.plan.restart_authority.operation
    assert payload["instance"]["consumer_cursors"]["rows"] == []
    assert pops.inspect(runtime) == payload


def test_checkpoint_graph_provider_is_the_resolved_restart_authority(tmp_path):
    plan, graph, manifest = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(bit_identical=True),
    )
    authority = plan.artifact.plan.restart_authority
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    assert authority.source == "consumer-graph"
    assert authority.operation is manifest.operation
    assert authority.to_data()["operation"] == dict(manifest.operation_data)
    assert runtime._restart_operation() is authority.operation
    assert graph.to_data()["identity"] == runtime.consumer_graph.to_data()["identity"]


def test_checkpoint_uses_explicit_precreated_inode_seam(monkeypatch, tmp_path):
    from pops.output._checkpoint_collective import write_precreated_checkpoint_payload
    from pops.runtime._checkpoint_manifest import seal_checkpoint_payload
    from pops.runtime._engine_descriptors import abi_key

    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._executor._last_run_identity = make_identity(
        "run", {"test": "checkpoint-precreated-inode-seam"}
    )
    evidence = {}

    def checkpoint_precreated(path, *, precreated_descriptor):
        assert type(precreated_descriptor) is int
        before = os.fstat(precreated_descriptor)
        payload = {
            "t": runtime._executor._time,
            "macro_step": runtime._executor._step,
            "abi_key": abi_key(),
        }
        seal_checkpoint_payload(runtime._executor, payload, runtime_kind="uniform")
        write_precreated_checkpoint_payload(precreated_descriptor, payload)
        after = os.fstat(precreated_descriptor)
        evidence.update(
            path=Path(path),
            owner=(int(before.st_dev), int(before.st_ino)),
            after=(int(after.st_dev), int(after.st_ino)),
        )
        return str(path)

    monkeypatch.setattr(
        runtime._executor,
        "_checkpoint_precreated_inode",
        checkpoint_precreated,
        raising=False,
    )

    assert runtime.checkpoint(tmp_path / "restart") == str(tmp_path / "restart.npz")
    assert evidence["owner"] == evidence["after"]
    assert not evidence["path"].exists()
    assert (tmp_path / "restart.npz").is_file()


def test_precreated_checkpoint_write_failure_closes_only_the_helper_owned_duplicate(
    monkeypatch, tmp_path
):
    from pops.output import _checkpoint_collective

    descriptor = os.open(tmp_path / "precreated.npz", os.O_CREAT | os.O_RDWR, 0o600)
    real_close = os.close
    real_dup = os.dup
    real_fdopen = os.fdopen
    duplicated = []
    explicit_closes = []
    fdopen_closefds = []

    def record_dup(source):
        duplicate = real_dup(source)
        duplicated.append(duplicate)
        return duplicate

    def record_close(target):
        explicit_closes.append(target)
        real_close(target)

    def record_fdopen(target, mode, *, closefd):
        fdopen_closefds.append(closefd)
        return real_fdopen(target, mode, closefd=closefd)

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("injected NPZ write failure")

    monkeypatch.setattr(_checkpoint_collective.os, "dup", record_dup)
    monkeypatch.setattr(_checkpoint_collective.os, "close", record_close)
    monkeypatch.setattr(_checkpoint_collective.os, "fdopen", record_fdopen)
    monkeypatch.setattr(np, "savez_compressed", fail_save)
    try:
        with pytest.raises(RuntimeError, match="injected NPZ write failure"):
            _checkpoint_collective.write_precreated_checkpoint_payload(
                descriptor, {"value": np.array([1], dtype=np.int64)}
            )

        assert len(duplicated) == 1
        assert fdopen_closefds == [False]
        assert explicit_closes == [duplicated[0]]
        with pytest.raises(OSError):
            os.fstat(duplicated[0])
        assert os.fstat(descriptor).st_ino > 0
    finally:
        real_close(descriptor)


def test_precreated_checkpoint_fdopen_failure_closes_only_the_owned_duplicate(monkeypatch, tmp_path):
    from pops.output import _checkpoint_collective

    descriptor = os.open(tmp_path / "precreated.npz", os.O_CREAT | os.O_RDWR, 0o600)
    real_close = os.close
    real_dup = os.dup
    duplicated = []
    explicit_closes = []

    def record_dup(source):
        duplicate = real_dup(source)
        duplicated.append(duplicate)
        return duplicate

    def record_close(target):
        explicit_closes.append(target)
        real_close(target)

    def fail_fdopen(target, mode, *, closefd):
        assert target == duplicated[0]
        assert mode == "wb"
        assert closefd is False
        raise RuntimeError("injected fdopen construction failure")

    monkeypatch.setattr(_checkpoint_collective.os, "dup", record_dup)
    monkeypatch.setattr(_checkpoint_collective.os, "close", record_close)
    monkeypatch.setattr(_checkpoint_collective.os, "fdopen", fail_fdopen)
    try:
        with pytest.raises(RuntimeError, match="injected fdopen construction failure"):
            _checkpoint_collective.write_precreated_checkpoint_payload(
                descriptor, {"value": np.array([1], dtype=np.int64)}
            )

        assert len(duplicated) == 1
        assert explicit_closes == [duplicated[0]]
        with pytest.raises(OSError):
            os.fstat(duplicated[0])
        assert os.fstat(descriptor).st_ino > 0
    finally:
        real_close(descriptor)


def test_checkpoint_reseal_failure_removes_its_owned_native_staging(monkeypatch, tmp_path):
    from pops.runtime import _checkpoint_manifest

    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._executor._last_run_identity = make_identity(
        "run", {"test": "checkpoint-reseal-cleanup"}
    )
    original_seal = _checkpoint_manifest.seal_checkpoint_payload
    calls = 0

    def fail_runtime_envelope(owner, payload, *, runtime_kind):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected RuntimeInstance envelope reseal failure")
        return original_seal(owner, payload, runtime_kind=runtime_kind)

    monkeypatch.setattr(_checkpoint_manifest, "seal_checkpoint_payload", fail_runtime_envelope)

    with pytest.raises(RuntimeError, match="injected RuntimeInstance envelope reseal failure"):
        runtime.checkpoint(tmp_path / "restart")

    assert calls == 2
    assert not (tmp_path / "restart.npz").exists()
    assert not tuple(tmp_path.glob(".pops-restart-transaction.*"))


def test_checkpoint_reseal_failure_never_deletes_a_replaced_staging_inode(monkeypatch, tmp_path):
    from pops.runtime import _checkpoint_manifest

    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._executor._last_run_identity = make_identity(
        "run", {"test": "checkpoint-reseal-replacement"}
    )
    original_seal = _checkpoint_manifest.seal_checkpoint_payload
    calls = 0
    replacement = b"third-party checkpoint staging replacement"
    evidence = {}

    def replace_staging_and_fail(owner, payload, *, runtime_kind):
        nonlocal calls
        calls += 1
        if calls == 2:
            (transaction,) = tuple(tmp_path.glob(".pops-restart-transaction.*"))
            staging = transaction / "native.npz"
            owned_inode = _path_owner(staging)
            third_party = tmp_path / "third-party-replacement.npz"
            third_party.write_bytes(replacement)
            replacement_inode = _path_owner(third_party)
            assert replacement_inode != owned_inode
            os.replace(third_party, staging)
            evidence.update(path=staging, inode=replacement_inode)
            raise RuntimeError("injected reseal failure after staging replacement")
        return original_seal(owner, payload, runtime_kind=runtime_kind)

    monkeypatch.setattr(
        _checkpoint_manifest,
        "seal_checkpoint_payload",
        replace_staging_and_fail,
    )

    with pytest.raises(
        RuntimeError,
        match="injected reseal failure after staging replacement",
    ) as caught:
        runtime.checkpoint(tmp_path / "restart")

    staging = evidence["path"]
    assert staging.read_bytes() == replacement
    assert _path_owner(staging) == evidence["inode"]
    assert any(
        "refuses to delete replaced transaction entry" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert not (tmp_path / "restart.npz").exists()


def test_checkpoint_refuses_path_only_capture_without_a_private_transaction_receipt(tmp_path):
    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    with pytest.raises(RuntimeError, match="path-only native ABI cannot prove creator ownership"):
        runtime._checkpoint_payload(tmp_path / "unreceipted")

    assert not (tmp_path / "unreceipted.npz").exists()


def test_checkpoint_replacement_before_entry_acquisition_is_never_cleaned(monkeypatch, tmp_path):
    from pops.runtime import _checkpoint_manifest

    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._executor._last_run_identity = make_identity(
        "run", {"test": "checkpoint-pre-acquisition-replacement"}
    )
    original_authenticate = _checkpoint_manifest.authenticate_checkpoint_payload
    replacement = b"third-party replacement before entry ownership"
    evidence = {}

    def replace_after_native_authentication(owner, payload, *, runtime_kind):
        identity = original_authenticate(owner, payload, runtime_kind=runtime_kind)
        (transaction,) = tuple(tmp_path.glob(".pops-restart-transaction.*"))
        staging = transaction / "native.npz"
        third_party = tmp_path / "third-party-before-acquisition.npz"
        third_party.write_bytes(replacement)
        os.replace(third_party, staging)
        evidence.update(path=staging, inode=_path_owner(staging))
        return identity

    monkeypatch.setattr(
        _checkpoint_manifest,
        "authenticate_checkpoint_payload",
        replace_after_native_authentication,
    )

    with pytest.raises(
        RuntimeError,
        match="replaced before ownership acquisition",
    ) as caught:
        runtime.checkpoint(tmp_path / "restart")

    staging = evidence["path"]
    assert staging.read_bytes() == replacement
    assert _path_owner(staging) == evidence["inode"]
    assert any(
        "transaction directory is not empty" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert not (tmp_path / "restart.npz").exists()


def test_checkpoint_never_reacquires_created_at_ownership_from_a_valid_replacement(
    monkeypatch, tmp_path
):
    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._executor._last_run_identity = make_identity(
        "run", {"test": "checkpoint-valid-native-replacement"}
    )
    original_checkpoint = runtime._executor.checkpoint
    evidence = {}

    def replace_valid_native_checkpoint(path):
        target = Path(original_checkpoint(path))
        payload = target.read_bytes()
        replacement = tmp_path / "valid-native-replacement.npz"
        replacement.write_bytes(payload)
        os.replace(replacement, target)
        evidence.update(path=target, payload=payload, owner=_path_owner(target))
        return str(target)

    monkeypatch.setattr(runtime._executor, "checkpoint", replace_valid_native_checkpoint)

    with pytest.raises(RuntimeError, match="replaced its created-at staging inode"):
        runtime.checkpoint(tmp_path / "restart")

    assert evidence["path"].read_bytes() == evidence["payload"]
    assert _path_owner(evidence["path"]) == evidence["owner"]
    assert not (tmp_path / "restart.npz").exists()


def test_checkpoint_eexist_same_inode_never_grants_expected_entry_ownership(monkeypatch, tmp_path):
    from pops.output._writers import common

    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._executor._last_run_identity = make_identity(
        "run", {"test": "checkpoint-eexist-same-inode"}
    )
    original_rename = common._rename_no_replace
    evidence = {}

    def create_same_inode_entry_before_rename(source, destination, *args, **kwargs):
        if destination == "native.npz" and source.endswith(".runtime-instance.tmp"):
            os.link(
                source,
                destination,
                src_dir_fd=kwargs["src_dir_fd"],
                dst_dir_fd=kwargs["dst_dir_fd"],
                follow_symlinks=False,
            )
            linked = os.stat(
                destination,
                dir_fd=kwargs["dst_dir_fd"],
                follow_symlinks=False,
            )
            evidence["inode"] = (int(linked.st_dev), int(linked.st_ino))
        return original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(common, "_rename_no_replace", create_same_inode_entry_before_rename)

    with pytest.raises(OSError, match="appeared during envelope publication") as caught:
        runtime.checkpoint(tmp_path / "restart")

    (transaction,) = tuple(tmp_path.glob(".pops-restart-transaction.*"))
    expected = transaction / "native.npz"
    assert _path_owner(expected) == evidence["inode"]
    assert not tuple(transaction.glob("*.runtime-instance.tmp"))
    assert any(
        "transaction directory is not empty" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert not (tmp_path / "restart.npz").exists()


def test_checkpoint_temporary_substitution_preserves_primary_error_and_replacement(
    monkeypatch, tmp_path
):
    from pops.output import _restart_provider

    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._executor._last_run_identity = make_identity(
        "run", {"test": "checkpoint-temporary-substitution"}
    )
    original_authenticate = _restart_provider._CheckpointTransactionReceipt.authenticate_entry_at
    replacement = b"third-party runtime envelope temporary"
    evidence = {}

    def replace_temporary_before_authentication(transaction, authority):
        if authority.name.endswith(".runtime-instance.tmp") and not evidence:
            third_party = tmp_path / "third-party-temporary.npz"
            third_party.write_bytes(replacement)
            temporary = transaction.directory / authority.name
            os.replace(third_party, temporary)
            evidence.update(
                path=temporary,
                inode=_path_owner(temporary),
            )
        return original_authenticate(transaction, authority)

    monkeypatch.setattr(
        _restart_provider._CheckpointTransactionReceipt,
        "authenticate_entry_at",
        replace_temporary_before_authentication,
    )

    with pytest.raises(
        RuntimeError,
        match="transaction entry was replaced before ownership acquisition",
    ) as caught:
        runtime.checkpoint(tmp_path / "restart")

    temporary = evidence["path"]
    assert temporary.read_bytes() == replacement
    assert _path_owner(temporary) == evidence["inode"]
    notes = getattr(caught.value, "__notes__", ())
    assert any("temporary cleanup" in note for note in notes)
    assert any("transaction directory is not empty" in note for note in notes)
    assert not (tmp_path / "restart.npz").exists()


def test_checkpoint_transaction_directory_substitution_never_uses_the_replacement(
    monkeypatch, tmp_path
):
    from pops.output import _restart_provider

    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._executor._last_run_identity = make_identity(
        "run", {"test": "checkpoint-transaction-directory-substitution"}
    )
    receipt_type = _restart_provider._CheckpointTransactionReceipt
    original_open = receipt_type.open_candidate_at
    replacement = b"third-party transaction directory"
    evidence = {}

    def substitute_directory(receipt, name):
        if not evidence:
            detached = tmp_path / "detached-owned-transaction"
            os.replace(receipt.directory, detached)
            receipt.directory.mkdir(mode=0o700)
            marker = receipt.directory / "third-party-marker"
            marker.write_bytes(replacement)
            evidence.update(detached=detached, marker=marker)
        return original_open(receipt, name)

    monkeypatch.setattr(receipt_type, "open_candidate_at", substitute_directory)

    with pytest.raises(RuntimeError, match="transaction directory authority changed"):
        runtime.checkpoint(tmp_path / "restart")

    assert evidence["marker"].read_bytes() == replacement
    assert evidence["detached"].is_dir()
    assert not (tmp_path / "restart.npz").exists()


def test_checkpoint_post_reseal_handoff_substitution_preserves_replacement(monkeypatch, tmp_path):
    from pops.output import _restart_provider

    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._executor._last_run_identity = make_identity(
        "run", {"test": "checkpoint-post-reseal-handoff-substitution"}
    )
    proof_type = _restart_provider._CheckpointPayloadProof
    original_to_data = proof_type.to_data
    replacement = b"third-party replacement after reseal"
    evidence = {}
    calls = 0

    def substitute_before_second_handoff(proof):
        nonlocal calls
        calls += 1
        if calls == 2:
            third_party = tmp_path / "third-party-after-reseal.npz"
            third_party.write_bytes(replacement)
            os.replace(third_party, proof.path)
            evidence.update(path=proof.path, owner=_path_owner(proof.path))
        return original_to_data(proof)

    monkeypatch.setattr(proof_type, "to_data", substitute_before_second_handoff)

    with pytest.raises(
        RuntimeError,
        match="transaction entry was replaced before ownership acquisition",
    ) as caught:
        runtime.checkpoint(tmp_path / "restart")

    assert calls == 2
    assert evidence["path"].read_bytes() == replacement
    assert _path_owner(evidence["path"]) == evidence["owner"]
    assert any(
        "rank-zero checkpoint cleanup also failed" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert not (tmp_path / "restart.npz").exists()


def test_checkpoint_resealed_descriptor_survives_handoff_until_rollback(tmp_path):
    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._executor._last_run_identity = make_identity(
        "run", {"test": "checkpoint-resealed-descriptor-lifecycle"}
    )
    operation = runtime._restart_operation()
    snapshot = operation.snapshot(runtime, tmp_path)
    proof = snapshot._proof
    assert proof is not None
    retained_fd = proof.entry.fileno()
    retained_owner = _path_owner(snapshot.path)
    assert (int(os.fstat(retained_fd).st_dev), int(os.fstat(retained_fd).st_ino)) == retained_owner

    target = operation.write(snapshot, tmp_path / "restart")

    assert target.is_file()
    assert snapshot._published_entry is not None
    assert snapshot._published_entry.fileno() == retained_fd
    assert (int(os.fstat(retained_fd).st_dev), int(os.fstat(retained_fd).st_ino)) == retained_owner
    snapshot.rollback()
    snapshot.rollback()
    snapshot.finalize()
    snapshot.finalize()
    assert not target.exists()
    with pytest.raises(OSError):
        os.fstat(retained_fd)


def test_checkpoint_discard_aggregates_independent_cleanup_failures(monkeypatch, tmp_path):
    from pops.output import _restart_provider

    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._executor._last_run_identity = make_identity(
        "run", {"test": "checkpoint-aggregate-discard-cleanup"}
    )
    snapshot = runtime._restart_operation().snapshot(runtime, tmp_path)
    receipt_type = _restart_provider._CheckpointTransactionReceipt
    calls = []

    def fail_quarantine(_receipt, entry, *, phase, close_entry=True):
        calls.append(("quarantine", phase))
        if close_entry:
            entry.close()
        raise RuntimeError("injected staging quarantine failure")

    def fail_directory_cleanup(receipt):
        calls.append(("directory", receipt.directory_name))
        receipt.close()
        raise RuntimeError("injected transaction directory cleanup failure")

    monkeypatch.setattr(receipt_type, "quarantine_entry_at", fail_quarantine)
    monkeypatch.setattr(receipt_type, "cleanup_empty", fail_directory_cleanup)

    with pytest.raises(RuntimeError, match="checkpoint snapshot cleanup failed") as caught:
        snapshot.discard()

    message = str(caught.value)
    assert "injected staging quarantine failure" in message
    assert "injected transaction directory cleanup failure" in message
    assert [row[0] for row in calls] == ["quarantine", "directory"]
    snapshot.finalize()
    snapshot.finalize()


def test_checkpoint_openat_refuses_symlink_entries_without_following(tmp_path):
    import errno
    import stat as stat_module

    from pops.output import _restart_provider

    receipt = _restart_provider._CheckpointTransactionReceipt.created(tmp_path)
    native = receipt.take_native_entry()
    os.symlink("missing-target", "candidate", dir_fd=receipt.directory_fileno())
    try:
        with pytest.raises(FileExistsError):
            receipt.created_at("candidate")
        with pytest.raises(OSError) as caught:
            receipt.open_candidate_at("candidate")
        assert caught.value.errno in {errno.ELOOP, errno.EMLINK}
        status = os.stat(
            "candidate",
            dir_fd=receipt.directory_fileno(),
            follow_symlinks=False,
        )
        assert stat_module.S_ISLNK(status.st_mode)
    finally:
        os.unlink("candidate", dir_fd=receipt.directory_fileno())
        receipt.quarantine_entry_at(native, phase="symlink refusal test cleanup")
        receipt.cleanup_empty()


def test_checkpoint_peer_proofs_are_opaque_scalars_without_rank_local_stat(monkeypatch, tmp_path):
    from pops.output import _restart_provider

    receipt_type = _restart_provider._CheckpointTransactionReceipt
    proof_type = _restart_provider._CheckpointPayloadProof
    receipt = receipt_type.created(tmp_path)
    receipt_data = receipt.to_data()
    native = receipt.take_native_entry()
    proof = proof_type(receipt, native)
    proof_data = proof.to_data()

    def forbid_rank_local_stat(*_args, **_kwargs):
        raise AssertionError("a peer compared root inode evidence with its local mount")

    with monkeypatch.context() as isolated:
        isolated.setattr(os, "stat", forbid_rank_local_stat)
        peer_receipt = receipt_type.observed(receipt_data)
        peer_proof = proof_type.observed(peer_receipt, proof_data)

    assert not peer_receipt.has_root_descriptor
    assert not peer_proof.entry.is_open
    assert peer_receipt.owner == receipt.owner
    assert peer_proof.owner == proof.owner
    receipt.quarantine_entry_at(native, phase="opaque peer proof test cleanup")
    receipt.cleanup_empty()


def test_checkpoint_root_attempt_broadcasts_exact_opaque_proof(monkeypatch):
    from pops.output import _checkpoint_collective

    communicator = object()
    topology = _checkpoint_collective.CheckpointTopology(0, 2, communicator)
    proof = {
        "path": "/opaque/root/path/native.npz",
        "entry_name": "native.npz",
        "entry_owner": [17, 23],
        "directory_name": ".pops-restart-transaction.test",
        "directory_owner": [5, 11],
    }
    envelopes = []

    def broadcast(actual_communicator, envelope, *, root):
        assert actual_communicator is communicator
        assert root == 0
        envelopes.append(envelope)
        return envelope

    monkeypatch.setattr(_checkpoint_collective, "broadcast_value", broadcast)

    attempt = _checkpoint_collective.root_attempt(topology, "proof handoff", lambda: proof)

    assert attempt.value == proof
    assert attempt.producer_error is None
    assert attempt.transport_error is None
    assert envelopes == [{"value": proof, "error": None}]


def test_checkpoint_root_attempt_keeps_producer_and_transport_failures_distinct(monkeypatch):
    from pops.output import _checkpoint_collective

    communicator = object()
    topology = _checkpoint_collective.CheckpointTopology(0, 2, communicator)
    producer_error = ValueError("injected producer failure")
    broadcasts = 0

    def fail_broadcast(_communicator, _envelope, *, root):
        nonlocal broadcasts
        assert root == 0
        broadcasts += 1
        raise OSError("injected transport failure")

    def fail_producer():
        raise producer_error

    monkeypatch.setattr(_checkpoint_collective, "broadcast_value", fail_broadcast)

    attempt = _checkpoint_collective.root_attempt(topology, "broken proof", fail_producer)

    assert broadcasts == 1
    assert attempt.producer_error is producer_error
    assert isinstance(attempt.transport_error, OSError)
    assert "injected transport failure" in str(attempt.transport_error)


def test_checkpoint_discard_transport_failure_performs_no_second_collective(monkeypatch, tmp_path):
    from pops.output import _checkpoint_collective, _restart_provider

    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._executor._last_run_identity = make_identity(
        "run", {"test": "checkpoint-discard-transport-failure"}
    )
    snapshot = runtime._restart_operation().snapshot(runtime, tmp_path)
    attempts = 0

    def break_after_root_cleanup(_topology, _phase, producer):
        nonlocal attempts
        attempts += 1
        producer()
        return _checkpoint_collective.RootAttempt(
            transport_error=OSError("injected post-cleanup transport failure")
        )

    monkeypatch.setattr(_checkpoint_collective, "root_attempt", break_after_root_cleanup)

    with pytest.raises(
        _restart_provider._CheckpointTransportFailure,
        match="transport failed during discard",
    ):
        snapshot.discard()

    assert attempts == 1
    assert not tuple(tmp_path.glob(".pops-restart-transaction.*"))


def test_checkpoint_reseal_fails_closed_when_atomic_quarantine_is_unavailable(
    monkeypatch, tmp_path
):
    from pops.output._writers import common

    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._executor._last_run_identity = make_identity(
        "run", {"test": "checkpoint-atomic-quarantine-unavailable"}
    )

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("injected atomic rename primitive unavailable")

    monkeypatch.setattr(common, "_rename_no_replace", unavailable)

    with pytest.raises(RuntimeError, match="atomic rename primitive unavailable") as caught:
        runtime.checkpoint(tmp_path / "restart")

    (transaction,) = tuple(tmp_path.glob(".pops-restart-transaction.*"))
    assert (transaction / "native.npz").is_file()
    assert tuple(transaction.glob("*.runtime-instance.tmp"))
    notes = getattr(caught.value, "__notes__", ())
    assert any("failed runtime checkpoint staging cleanup" in note for note in notes)
    assert any("transaction directory is not empty" in note for note in notes)
    assert not (tmp_path / "restart.npz").exists()


def test_runtime_instance_has_one_authored_execution_route():
    plan = _install()
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    declared_public = {name for name in RuntimeInstance.__dict__ if not name.startswith("_")}
    assert {name for name in dir(runtime) if not name.startswith("_")} == declared_public
    assert not hasattr(runtime, "__dict__")
    with pytest.raises(AttributeError):
        runtime.engine = object()
    assert not hasattr(runtime, "step")
    assert not hasattr(runtime, "step_cfl")
    assert not hasattr(runtime, "run")
    assert not hasattr(runtime, "native_executor")
    assert not hasattr(runtime, "executor_for_layout")
    assert not hasattr(runtime, "executor_for_block")
    assert not hasattr(runtime, "install_plan")
    assert not hasattr(runtime, "runtime_plan")
    assert not hasattr(runtime, "assembly")
    assert not hasattr(runtime, "profile")
    assert not hasattr(runtime, "an_arbitrary_native_method")


def test_runtime_instance_refuses_ambiguous_global_state_without_provider_capability():
    class _LevelExplicitExecutor(_Executor):
        state_global = None

        def block_level_state_global(self, block, level):
            assert (block, level) == ("fluid", 0)
            return np.full(self._nx * self._ny, 3.0)

    plan = _install()
    runtime = RuntimeInstance(plan, executor=_LevelExplicitExecutor(plan))

    with pytest.raises(NotImplementedError, match="block_level_state_global"):
        runtime.state_global("fluid")
    assert np.all(runtime.block_level_state_global("fluid", 0) == 3.0)


def test_uniform_runtime_instance_exposes_one_level_without_an_amr_provider():
    plan = _install()
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    assert runtime.n_levels() == 1


@pytest.mark.parametrize(
    "controls",
    [
        {"strategy": FixedDt(1.0), "unknown_control": True},
        {"cfl": 0.4, "unknown_control": True},
        {"strategy": FixedDt(1.0), "cfl": 0.4, "unknown_control": True},
    ],
)
def test_runtime_engine_rejects_public_strategy_controls(controls):
    plan = _install()
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    with pytest.raises(TypeError, match="does not accept strategy= or cfl="):
        runtime._run(t_end=1.0, max_steps=1, **controls)


def test_consumer_moment_uses_the_accepted_qualified_child_clock_cursor(tmp_path):
    plan, _, manifest = _with_graph(tmp_path)
    executor = _Executor(plan)
    runtime = RuntimeInstance(plan, executor=executor)
    child = manifest.schedule.domain.clock
    macro = Clock("macro", owner=child.owner)
    temporal = TemporalRestartState()
    temporal.configure_program(
        {
            "schema_version": 1,
            "kind": "pops.temporal-program-schedule",
            "primary_clock": macro.qualified_id,
            "clocks": [
                {"id": macro.qualified_id, "descriptor": macro.to_data(), "ticks_per_macro": 1},
                {"id": child.qualified_id, "descriptor": child.to_data(), "ticks_per_macro": 4},
            ],
            "subcycles": [
                {
                    "node_id": 3,
                    "parent_clock": macro.qualified_id,
                    "child_clock": child.qualified_id,
                    "count": 4,
                }
            ],
            "synchronizations": [],
            "schedules": [],
            "histories": [],
        },
        time=0.0,
        macro_step=0,
    )
    executor._temporal_restart_state = temporal
    executor._time, executor._step = 0.25, 1
    temporal.accept(before_time=0.0, before_step=0, time=0.25, macro_step=1)

    (moment,) = runtime._moments()
    assert moment.point.step == 4
    assert moment.clock_tick == 4
    assert moment.physical_time_hex == (0.25).hex()
    assert moment.accepted_step == 1 and moment.wall_tick == 1


def test_consumer_moment_refuses_an_absent_qualified_clock(tmp_path):
    plan, _, _ = _with_graph(tmp_path)
    executor = _Executor(plan)
    runtime = RuntimeInstance(plan, executor=executor)
    unrelated = Clock("unrelated")
    temporal = TemporalRestartState()
    temporal.configure_program(
        {
            "schema_version": 1,
            "kind": "pops.temporal-program-schedule",
            "primary_clock": unrelated.qualified_id,
            "clocks": [
                {
                    "id": unrelated.qualified_id,
                    "descriptor": unrelated.to_data(),
                    "ticks_per_macro": 1,
                }
            ],
            "subcycles": [],
            "synchronizations": [],
            "schedules": [],
            "histories": [],
        },
        time=0.0,
        macro_step=0,
    )
    executor._temporal_restart_state = temporal

    with pytest.raises(RuntimeError, match="no cursor for qualified clock"):
        runtime._moments()


def test_run_publishes_exact_npz_only_after_accepted_step_and_commits_cursor(tmp_path):
    plan, graph, manifest = _with_graph(tmp_path)
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    report = runtime._run(t_end=1.0, max_steps=1)
    assert report.accepted_steps == 1

    cursor = runtime.consumer_cursors.for_consumer(manifest.qualified_id)
    assert cursor.committed_samples == 1
    outputs = tuple(tmp_path.glob("*.npz"))
    assert len(outputs) == 1
    reopened = read_npz(outputs[0])
    assert reopened.manifest["snapshot"]["clock"]["macro_step"] == 1
    assert reopened.manifest["snapshot"]["metadata"] == {
        "consumer_graph": graph.identity.token,
        "runtime_plan": runtime._runtime_plan.identity.token,
    }


class _BlockingWriterSession:
    def __init__(self, owner, snapshot, request, target):
        from pops.output import writer_session_authority

        self.authority = writer_session_authority("blocking-test", request, target)
        self.identity = Identity.from_token(self.authority["session_identity"])
        self._owner = owner
        self._snapshot = snapshot
        self._request = request
        self._target = Path(target)
        self._published = False

    def stage(self):
        self._owner.writer_started.set()
        if not self._owner.release_writer.wait(timeout=10):
            raise TimeoutError("test writer was not released")

    def abort_prepare(self):
        return None

    def publish(self):
        self._target.parent.mkdir(parents=True, exist_ok=True)
        self._target.write_text("exact async artifact\n")
        self._published = True
        self._owner.paths.append(self._target)
        return OutputPublicationReceipt(
            self._target,
            "blocking-test",
            make_identity(
                "scientific-output",
                {
                    "selection": self._request.publication_identity.token,
                },
            ),
            self._request.publication_identity,
        )

    def rollback(self):
        self._target.unlink(missing_ok=True)

    def finalize(self):
        if not self._published:
            raise RuntimeError("blocking writer finalized before publication")
        return None


class _BlockingWriter:
    format = "blocking-test"

    def __init__(self, owner):
        self._owner = owner

    def preflight(self, _execution_context):
        return {"schema_version": 1, "provider_id": "blocking-test", "serial": True}

    def prepare_session(self, snapshot, request, target, *, communicator=None):
        assert communicator is None
        return _BlockingWriterSession(self._owner, snapshot, request, target)


class _BlockingFormat:
    __pops_ir_immutable__ = True

    def __init__(self, mode: ParallelMode):
        self._mode = mode
        self.writer_started = threading.Event()
        self.release_writer = threading.Event()
        self.paths = []

    def consumer_data(self):
        return {
            "schema_version": 1,
            "provider_id": "pops.test.blocking-async.v1",
            "format_name": "blocking-test",
            "extension": ".async",
            "parallel_mode": self._mode.value,
        }

    def writer(self):
        return _BlockingWriter(self)


def test_async_scientific_output_overlaps_next_step_and_flushes_real_receipts(tmp_path):
    output_root = tmp_path / "async-output"
    output_root.mkdir()
    format_provider = _BlockingFormat(_scientific_output_mode(_install().artifact))
    authoring_clock = Clock("async-authoring")
    descriptor = AsyncScientificOutput(
        format=format_provider,
        schedule=Schedule(Every(AcceptedStep(authoring_clock), 1)),
        fields=(Handle("rho", kind="state", owner=OwnerPath.model("async-authoring")),),
        target="async-output",
        queue_capacity=1,
    )
    operation = descriptor.consumer_authoring()[0].operation
    plan, _, _ = _with_graph(
        output_root,
        kind=ConsumerKind.MONITOR,
        output_format=None,
        operation=operation,
    )

    class _ProgressExecutor(_Executor):
        def __init__(self, install):
            super().__init__(install)
            self.second_step_finalized = threading.Event()

        def _finalize_step_transaction(self):
            super()._finalize_step_transaction()
            if self._step >= 2:
                self.second_step_finalized.set()

    executor = _ProgressExecutor(plan)
    runtime = RuntimeInstance(plan, executor=executor)
    results = []
    errors = []

    def run():
        try:
            results.append(runtime._run(t_end=2.0, max_steps=2))
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run, daemon=False)
    worker.start()
    assert format_provider.writer_started.wait(timeout=5)
    assert executor.second_step_finalized.wait(timeout=5)
    assert worker.is_alive(), "end-of-run flush must still wait for the blocked writer"
    format_provider.release_writer.set()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert errors == []
    assert results[0].accepted_steps == 2
    assert len(runtime.post_commit_reports) == 2
    assert all(report.status == "delivered" for report in runtime.post_commit_reports)
    assert len(format_provider.paths) == 2
    assert all(path.is_file() for path in format_provider.paths)
    assert {
        Path(report.receipt.detail["path"])
        for report in runtime.post_commit_reports
        if report.receipt is not None
    } == set(format_provider.paths)


class _FailingPostCommitSession:
    authority = {
        "schema_version": 1,
        "provider_id": "pops.test.failing-post-commit.v1",
        "delivery": "post_commit",
        "threading": "dedicated_serial",
        "worker_mpi": False,
    }

    def initialize(self, _run):
        return None

    def execute(self, _frame):
        raise RuntimeError("viewer is unavailable")

    def finalize(self):
        return None

    def abort(self):
        return None


class _FailingPostCommitProvider:
    def consumer_data(self):
        return {
            "schema_version": 1,
            "provider_id": "pops.test.failing-post-commit.v1",
            "observer_kind": "test",
        }

    def open_session(self, _execution_context):
        return _FailingPostCommitSession()


@pytest.mark.parametrize(
    ("policy", "raises"),
    ((ReportOnly(), False), (RaiseOnFlush(), True)),
)
def test_post_commit_failure_policy_is_applied_only_at_run_flush(
    tmp_path,
    policy,
    raises,
):
    descriptor = LiveVisualization(
        observer=_FailingPostCommitProvider(),
        schedule=Schedule(Every(AcceptedStep(Clock("live-authoring")), 1)),
        fields=(Handle("rho", kind="state", owner=OwnerPath.model("live-authoring")),),
        on_failure=policy,
    )
    operation = descriptor.consumer_authoring()[0].operation
    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.MONITOR,
        output_format=None,
        operation=operation,
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    if raises:
        with pytest.raises(RuntimeError, match="post-commit consumer delivery failed"):
            runtime._run(t_end=1.0, max_steps=1)
    else:
        runtime._run(t_end=1.0, max_steps=1)

    assert runtime.time() == 1.0
    assert len(runtime.post_commit_reports) == 1
    assert runtime.post_commit_reports[0].status == "skipped"


class _OpenFailureProvider(_FailingPostCommitProvider):
    def open_session(self, _execution_context):
        raise RuntimeError("optional visualization dependency is missing")


def test_post_commit_session_dependency_failure_is_refused_before_any_step(tmp_path):
    descriptor = LiveVisualization(
        observer=_OpenFailureProvider(),
        schedule=Schedule(Every(AcceptedStep(Clock("preflight-authoring")), 1)),
        fields=(Handle("rho", kind="state", owner=OwnerPath.model("preflight-authoring")),),
    )
    operation = descriptor.consumer_authoring()[0].operation
    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.MONITOR,
        output_format=None,
        operation=operation,
    )
    executor = _Executor(plan)

    with pytest.raises(RuntimeError, match="session preflight failed"):
        RuntimeInstance(plan, executor=executor)

    assert executor.macro_step() == 0


class _InitializeFailureSession(_FailingPostCommitSession):
    authority = {
        **_FailingPostCommitSession.authority,
        "provider_id": "pops.test.initialize-failure.v1",
    }

    def __init__(self):
        self.abort_calls = 0

    def initialize(self, _run):
        raise RuntimeError("run-scoped observer initialization failed")

    def abort(self):
        self.abort_calls += 1


class _InitializeFailureProvider:
    def __init__(self):
        self.session = _InitializeFailureSession()

    def consumer_data(self):
        return {
            "schema_version": 1,
            "provider_id": "pops.test.initialize-failure.v1",
            "observer_kind": "test",
        }

    def open_session(self, _execution_context):
        return self.session


def test_post_commit_run_initialization_failure_precedes_start_sample_and_first_step(tmp_path):
    provider = _InitializeFailureProvider()
    descriptor = LiveVisualization(
        observer=provider,
        schedule=Schedule(Every(AcceptedStep(Clock("initialize-failure")), 1)),
        fields=(Handle("rho", kind="state", owner=OwnerPath.model("initialize-failure")),),
    )
    operation = descriptor.consumer_authoring()[0].operation
    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.MONITOR,
        output_format=None,
        operation=operation,
    )
    executor = _Executor(plan)
    runtime = RuntimeInstance(plan, executor=executor)

    with pytest.raises(RuntimeError, match="session initialization failed"):
        runtime._run(t_end=1.0, max_steps=1)

    assert executor.macro_step() == 0
    assert provider.session.abort_calls == 1


class _InjectedFinalizeDiagnosticSession(_FailingPostCommitSession):
    authority = {
        **_FailingPostCommitSession.authority,
        "provider_id": "pops.test.injected-finalize-diagnostic.v1",
    }

    def execute(self, frame):
        return ObserverReceipt(
            frame.identity,
            self.authority["provider_id"],
            {"writer_finalize_error": "not an async-writer diagnostic"},
        )


class _InjectedFinalizeDiagnosticProvider:
    def consumer_data(self):
        return {
            "schema_version": 1,
            "provider_id": "pops.test.injected-finalize-diagnostic.v1",
            "observer_kind": "test",
        }

    def open_session(self, _execution_context):
        return _InjectedFinalizeDiagnosticSession()


def test_generic_observer_cannot_inject_async_writer_finalize_failure(tmp_path):
    descriptor = LiveVisualization(
        observer=_InjectedFinalizeDiagnosticProvider(),
        schedule=Schedule(Every(AcceptedStep(Clock("injected-diagnostic")), 1)),
        fields=(Handle("rho", kind="state", owner=OwnerPath.model("injected-diagnostic")),),
        on_failure=RaiseOnFlush(),
    )
    operation = descriptor.consumer_authoring()[0].operation
    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.MONITOR,
        output_format=None,
        operation=operation,
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    report = runtime._run(t_end=1.0, max_steps=1)

    assert report.accepted_steps == 1
    assert runtime.post_commit_reports[0].status == "delivered"
    assert runtime.post_commit_diagnostics == ()


def _published_times(root: Path) -> list[float]:
    return sorted(
        float.fromhex(read_npz(path).manifest["snapshot"]["clock"]["time"])
        for path in root.rglob("*.npz")
    )


def test_every_dt_clips_adaptive_steps_to_exact_thresholds_without_end_duplicate(tmp_path):
    output_root = tmp_path / "adaptive-outputs"
    plan, _, manifest = _with_graph(
        output_root,
        schedule=lambda clock: every_dt(0.25, clock=clock),
    )

    class _AdaptiveExecutor(_Executor):
        def step_cfl(self, cfl, *, max_dt, min_dt):
            assert cfl == pytest.approx(0.4)
            dt = min(0.4, float(max_dt))
            if dt < float(min_dt):
                raise RuntimeError("test adaptive stability bound is below min_dt")
            self.step(dt)
            return dt

    executor = _AdaptiveExecutor(plan)
    executor._step_strategy = AdaptiveCFL(0.4)
    runtime = RuntimeInstance(plan, executor=executor)

    report = runtime._run(t_end=0.5, max_steps=2)

    assert report.accepted_steps == 2
    assert runtime.time() == 0.5
    assert _published_times(output_root) == [0.25, 0.5]
    assert runtime.consumer_cursors.for_consumer(manifest.qualified_id).committed_samples == 2


def test_every_dt_restart_rederives_next_deadline_without_republishing_boundary(tmp_path):
    output_root = tmp_path / "restart-outputs"
    plan, _, manifest = _with_graph(
        output_root,
        schedule=lambda clock: every_dt(0.25, clock=clock),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._run(t_end=0.5, max_steps=2)
    source_temporal = runtime._executor._temporal_restart_state.to_data()
    checkpoint = runtime.checkpoint(tmp_path / "physical-cadence-restart")

    restored = RuntimeInstance(plan, executor=_Executor(plan))
    restored.restart(checkpoint)
    restored_temporal = restored._executor._temporal_restart_state
    assert restored_temporal.time_hex == (0.5).hex()
    assert restored_temporal.macro_step == 2
    assert restored_temporal.to_data() == source_temporal
    report = restored._run(t_end=0.75, max_steps=1)

    assert report.accepted_steps == 1
    assert restored.time() == 0.75
    assert _published_times(output_root) == [0.25, 0.5, 0.75]
    assert restored.consumer_cursors.for_consumer(manifest.qualified_id).committed_samples == 3


@pytest.mark.parametrize(
    ("interval", "t_end", "grid"),
    (
        (0.25, 0.5, (0.0, 0.2, 0.5)),
        (1.0e-20, 2.0e-20, (0.0, 2.0e-20)),
    ),
)
def test_every_dt_requires_each_active_deadline_in_external_time_grid(
    tmp_path,
    interval,
    t_end,
    grid,
):
    output_root = tmp_path / "external-grid-outputs"
    plan, _, _ = _with_graph(
        output_root,
        schedule=lambda clock: every_dt(interval, clock=clock),
    )
    executor = _Executor(plan)
    executor._step_strategy = ExternalTimeGrid("forcing_times")
    runtime = RuntimeInstance(plan, executor=executor)

    with pytest.raises(ValueError, match="absent from ExternalTimeGrid"):
        runtime._run(
            t_end=t_end,
            max_steps=2,
            forcing_times=grid,
        )

    assert runtime.time() == 0.0
    assert _published_times(output_root) == []


def test_every_dt_accepts_equivalent_external_grid_threshold_rounding(tmp_path):
    output_root = tmp_path / "compatible-external-grid-outputs"
    plan, _, manifest = _with_graph(
        output_root,
        schedule=lambda clock: every_dt(0.1, clock=clock),
    )
    executor = _Executor(plan)
    executor._step_strategy = ExternalTimeGrid("forcing_times")
    runtime = RuntimeInstance(plan, executor=executor)

    report = runtime._run(
        t_end=0.3,
        max_steps=3,
        forcing_times=(0.0, 0.1, 0.2, 0.3),
    )

    assert report.accepted_steps == 3
    assert runtime.time() == 0.3
    assert _published_times(output_root) == [0.1, 0.2, 0.3]
    assert runtime.consumer_cursors.for_consumer(manifest.qualified_id).committed_samples == 3


def test_every_dt_external_grid_must_not_land_before_threshold(tmp_path):
    output_root = tmp_path / "early-external-grid-output"
    plan, _, _ = _with_graph(
        output_root,
        schedule=lambda clock: every_dt(0.1, clock=clock),
    )
    executor = _Executor(plan)
    executor._step_strategy = ExternalTimeGrid("forcing_times")
    runtime = RuntimeInstance(plan, executor=executor)

    with pytest.raises(ValueError, match="absent from ExternalTimeGrid"):
        runtime._run(
            t_end=0.1,
            max_steps=1,
            forcing_times=(0.0, np.nextafter(0.1, -np.inf).item()),
        )

    assert runtime.time() == 0.0
    assert _published_times(output_root) == []


def test_every_dt_merges_equivalent_run_end_without_duplicate_micro_step(tmp_path):
    output_root = tmp_path / "merged-run-end-output"
    t_end = 3.0 * 0.1
    assert t_end == np.nextafter(0.3, np.inf)
    plan, _, manifest = _with_graph(
        output_root,
        schedule=lambda clock: every_dt(0.1, clock=clock),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    report = runtime._run(t_end=t_end, max_steps=3)

    assert report.accepted_steps == 3
    assert runtime.time() == t_end
    assert _published_times(output_root) == [0.1, 0.2, t_end]
    assert runtime.consumer_cursors.for_consumer(manifest.qualified_id).committed_samples == 3


def test_every_dt_threshold_one_ulp_after_run_end_is_not_due(tmp_path):
    output_root = tmp_path / "nextafter-output"
    interval = np.nextafter(0.1, np.inf).item()
    plan, _, manifest = _with_graph(
        output_root,
        schedule=lambda clock: every_dt(interval, clock=clock),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    report = runtime._run(t_end=0.3, max_steps=3)

    assert report.accepted_steps == 3
    assert runtime.time() == 0.3
    assert _published_times(output_root) == [interval, 2.0 * interval]
    assert 3.0 * interval == np.nextafter(0.3, np.inf)
    assert runtime.consumer_cursors.for_consumer(manifest.qualified_id).committed_samples == 2


def test_run_fails_explicitly_when_max_steps_cannot_reach_t_end(tmp_path):
    plan, _, manifest = _with_graph(
        tmp_path, schedule=lambda clock: Schedule(AtEnd(AcceptedStep(clock)))
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    with pytest.raises(RuntimeError, match="max_steps exhausted before t_end"):
        runtime._run(2.0, max_steps=1)

    assert runtime.time() == 1.0
    cursor = runtime.consumer_cursors.for_consumer(manifest.qualified_id)
    assert cursor.committed_samples == 0
    assert tuple(tmp_path.glob("*.npz")) == ()


def test_failed_run_keeps_identity_sealed_when_entry_rollback_fails():
    class _RollbackFailureExecutor(_Executor):
        def _restore_temporal_restart_state(self, _state):
            raise RuntimeError("injected run-entry rollback failure")

    plan = _install()
    runtime = RuntimeInstance(plan, executor=_RollbackFailureExecutor(plan))
    calls = []
    close_failed = runtime._publisher.close_failed_run_consumers

    def capture_close(run_identity, *, release_identity, entry_effect_fence=None):
        calls.append((run_identity, release_identity))
        return close_failed(
            run_identity,
            release_identity=release_identity,
            entry_effect_fence=entry_effect_fence,
        )

    runtime._publisher.close_failed_run_consumers = capture_close
    with pytest.raises(RuntimeError, match="max_steps exhausted") as caught:
        runtime._run(t_end=1.0, max_steps=0, console=False)

    assert "injected run-entry rollback failure" in "\n".join(caught.value.__notes__)
    assert len(calls) == 1
    run_identity, release_identity = calls[0]
    assert release_identity is False
    assert run_identity.token in runtime._publisher._closed_observer_runs


def test_runtime_world_collective_loss_skips_post_commit_cleanup_and_stays_sealed(
    monkeypatch, request
):
    from pops.runtime import _runtime_consumers
    from pops.runtime._observer_runtime import PostCommitObserverWorker

    plan = _install()
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    publisher = runtime._publisher
    original_begin = publisher.begin_post_commit_consumers
    cleanup_calls = []
    workers = []

    def cleanup_workers():
        for worker in workers:
            if worker.close_succeeded is not True:
                worker.seal_local(RuntimeError("test cleanup"))

    request.addfinalizer(cleanup_workers)

    def lose_world(run_identity):
        worker = PostCommitObserverWorker(
            thread_name="test-runtime-world-loss-local-seal",
            run_identity=run_identity,
        )
        workers.append(worker)
        publisher._observer_workers[run_identity.token] = worker
        raise _runtime_consumers._ObserverCollectiveLost(
            "injected runtime MPI_COMM_WORLD proof loss"
        )

    def forbidden_cleanup(*_args, **_kwargs):
        cleanup_calls.append(True)
        raise AssertionError("WORLD loss must skip post-commit cleanup")

    publisher.begin_post_commit_consumers = lose_world
    publisher.close_failed_run_consumers = forbidden_cleanup
    publisher.close_live_visualizations = forbidden_cleanup

    with pytest.raises(
        _runtime_consumers._ObserverCollectiveLost,
        match="injected runtime MPI_COMM_WORLD proof loss",
    ) as caught:
        runtime._run(t_end=0.0, max_steps=0, console=False)

    assert cleanup_calls == []
    assert len(workers) == 1
    assert workers[0].close_succeeded is True
    assert publisher._observer_workers[runtime.last_run_identity.token] is workers[0]
    assert "cleanup was skipped" in "\n".join(caught.value.__notes__)
    assert "injected runtime MPI_COMM_WORLD proof loss" in (
        publisher._observer_world_collective_lost
    )
    assert publisher.seal_observer_collective_loss(RuntimeError("later local refusal")) is True

    publisher.begin_post_commit_consumers = original_begin
    monkeypatch.setattr(
        RuntimeInstance,
        "_step_transaction_methods",
        lambda self: pytest.fail(
            "a sealed observer WORLD must refuse before native run preparation"
        ),
    )
    with pytest.raises(RuntimeError, match="MPI_COMM_WORLD is sealed"):
        runtime._run(t_end=0.0, max_steps=0, console=False)
    assert cleanup_calls == []


@pytest.mark.parametrize(
    "schedule",
    (
        lambda clock: Schedule(Always(AcceptedStep(clock))),
        lambda clock: Schedule(Every(AcceptedStep(clock), 1)),
        lambda clock: Schedule(AtEnd(AcceptedStep(clock))),
        lambda clock: Schedule(When(AcceptedStep(clock), True)),
    ),
)
def test_zero_step_run_does_not_fabricate_an_accepted_consumer_occurrence(tmp_path, schedule):
    plan, _, manifest = _with_graph(tmp_path, schedule=schedule)
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    report = runtime._run(t_end=0.0, max_steps=0)

    assert report.accepted_steps == 0
    assert runtime.consumer_cursors.for_consumer(manifest.qualified_id).committed_samples == 0
    assert tuple(tmp_path.glob("*.npz")) == ()


def test_zero_step_run_keeps_exactly_one_start_occurrence(tmp_path):
    plan, _, manifest = _with_graph(
        tmp_path,
        schedule=lambda clock: Schedule(AtStart(AcceptedStep(clock))),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    report = runtime._run(t_end=0.0, max_steps=0)

    assert report.accepted_steps == 0
    assert runtime.consumer_cursors.for_consumer(manifest.qualified_id).committed_samples == 1
    assert _published_times(tmp_path) == [0.0]


def test_scientific_format_is_a_structural_provider_without_name_dispatch(tmp_path):
    plan, _, _ = _with_graph(tmp_path, output_format=_CustomNPZ)
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    runtime._run(t_end=1.0, max_steps=1)

    assert len(tuple(tmp_path.glob("*.npz"))) == 1


def test_malformed_format_provider_is_refused_before_an_effect_exists(tmp_path):
    class _Malformed:
        __pops_ir_immutable__ = True

        def consumer_data(self):
            return {"schema_version": 1}

        def writer(self):
            return object()

    with pytest.raises((TypeError, ValueError), match="provider|writer|keys"):
        _with_graph(tmp_path, output_format=_Malformed())


def test_checkpoint_provider_requires_a_compensatable_snapshot_protocol(tmp_path):
    class _MalformedCheckpoint:
        __pops_ir_immutable__ = True

        @staticmethod
        def consumer_data():
            return {
                "schema_version": 1,
                "provider_id": "pops.test.malformed-checkpoint",
                "extension": ".npz",
            }

        @staticmethod
        def snapshot(_runtime, _directory):
            return object()

        @staticmethod
        def write(_snapshot, _target):
            raise AssertionError("unreachable")

        @staticmethod
        def reopen(_runtime, _path):
            raise AssertionError("unreachable")

        @staticmethod
        def restore(_runtime, _reopened):
            raise AssertionError("unreachable")

    with pytest.raises(TypeError, match="validate_snapshot"):
        _with_graph(
            tmp_path,
            kind=ConsumerKind.CHECKPOINT,
            output_format=None,
            operation=_MalformedCheckpoint(),
        )
    with pytest.raises(TypeError, match="discard/rollback"):
        RestartV3().validate_snapshot(object())


def test_checkpoint_restore_invalidates_geometry_after_native_topology_restore(monkeypatch):
    from pops.output import _checkpoint_collective

    events = []
    source_run_identity = make_identity("run", {"test": "restart-source"})

    class _Publisher:
        @staticmethod
        def validate_diagnostic_restart_state(data):
            assert data == {"schema_version": 1}
            return ("canonical",)

        @staticmethod
        def restore_diagnostic_restart_state(data):
            assert data == ("canonical",)
            events.append("diagnostics")

        @staticmethod
        def diagnostic_restart_state():
            return ("before",)

    class _SnapshotBuilder:
        def __init__(self):
            self._geometry_cache = {}

        @staticmethod
        def invalidate_geometry_cache():
            events.append("geometry")

    def restore_checkpoint_payload(
        runtime,
        executor,
        payload,
        *,
        bit_identical,
        hierarchy_mode,
        hierarchy_identity,
        phase_prefix,
        prepare_outer_state,
        after_native_apply,
        rollback_after_native_apply,
    ):
        assert runtime is owner and executor is native
        assert payload == b"checkpoint"
        assert bit_identical is True
        assert hierarchy_mode == "restore_recorded_hierarchy"
        assert hierarchy_identity is None
        assert phase_prefix == "native restart"
        assert callable(prepare_outer_state)
        assert callable(after_native_apply)
        assert callable(rollback_after_native_apply)
        prepare_outer_state()
        events.append("native")
        result = "restored"
        after_native_apply(result)
        return result

    class _Native:
        def __init__(self):
            self._last_run_manifest = None
            self._last_run_identity = None
            self._restart_lineage_identity = None

        @staticmethod
        def _restore_checkpoint_run_identity(identity):
            assert identity == source_run_identity
            events.append("run")

    native = _Native()
    from pops.runtime._checkpoint_resource_budget import _producer_checkpoint_resource_budget

    resource_budget = _producer_checkpoint_resource_budget(
        {"checkpoint": np.frombuffer(b"checkpoint", dtype=np.uint8)},
        runtime_kind="uniform",
        authority="test-runtime-restore",
    )
    monkeypatch.setattr(
        _checkpoint_collective,
        "decode_checkpoint_bytes",
        lambda _payload, _budget: {
            "runtime_consumer_diagnostics": np.array(json.dumps({"schema_version": 1}))
        },
    )
    monkeypatch.setattr(
        _checkpoint_collective, "restore_checkpoint_payload", restore_checkpoint_payload
    )
    monkeypatch.setattr(
        "pops.runtime._checkpoint_manifest.checkpoint_run_identity",
        lambda payload: source_run_identity,
    )
    cursors = ConsumerCursorSet()
    owner = SimpleNamespace(
        _executor=native,
        _snapshot_builder=_SnapshotBuilder(),
        _publisher=_Publisher(),
        _consumer_cursors=None,
        _checkpoint_resource_budget=resource_budget,
    )

    assert (
        RuntimeInstance._restore_checkpoint(
            owner,
            b"checkpoint",
            cursors,
            bit_identical=True,
        )
        == "restored"
    )
    assert owner._consumer_cursors is cursors
    assert events == ["native", "geometry", "diagnostics", "run"]


def test_checkpoint_restore_requires_exact_bit_identical_policy():
    owner = SimpleNamespace()

    with pytest.raises(
        TypeError, match="RuntimeInstance restart bit_identical must be an exact bool"
    ):
        RuntimeInstance._restore_checkpoint(
            owner,
            b"checkpoint",
            ConsumerCursorSet(),
            bit_identical=1,
        )


def test_restart_provider_passes_bit_identical_policy_without_hidden_state():
    calls = []

    class _Runtime:
        @staticmethod
        def _restore_checkpoint(payload, cursors, *, bit_identical):
            calls.append((payload, cursors, bit_identical))
            return "restored"

    cursors = ConsumerCursorSet()
    reopened = ReopenedRestart(Path("checkpoint.npz"), b"checkpoint", cursors)

    assert RestartV3(bit_identical=True).restore(_Runtime(), reopened) == "restored"
    assert calls == [(b"checkpoint", cursors, True)]


def test_restart_provider_passes_regrid_policy_identity_explicitly():
    from pops.output import RegridOnRestart

    calls = []
    hierarchy = RegridOnRestart()

    class _Runtime:
        @staticmethod
        def _restore_checkpoint(
            payload,
            cursors,
            *,
            bit_identical,
            hierarchy_mode,
            hierarchy_identity,
        ):
            calls.append(
                (
                    payload,
                    cursors,
                    bit_identical,
                    hierarchy_mode,
                    hierarchy_identity,
                )
            )
            return "restored"

    cursors = ConsumerCursorSet()
    reopened = ReopenedRestart(Path("checkpoint.npz"), b"checkpoint", cursors)

    assert RestartV3(hierarchy=hierarchy).restore(_Runtime(), reopened) == "restored"
    assert calls == [
        (
            b"checkpoint",
            cursors,
            False,
            "regrid_on_restart",
            hierarchy.identity.token,
        )
    ]


def test_collective_restart_passes_policy_to_exact_native_preflight():
    from pops.output._checkpoint_collective import restore_checkpoint_payload

    calls = []

    class _Executor:
        @staticmethod
        def _prepare_checkpoint_restart(payload, *, bit_identical):
            calls.append(("prepare", payload, bit_identical))
            return "prepared"

        @staticmethod
        def _begin_checkpoint_restart():
            calls.append(("begin",))

        @staticmethod
        def _apply_checkpoint_restart(prepared):
            calls.append(("apply", prepared))
            return "restored"

        @staticmethod
        def _commit_checkpoint_restart():
            calls.append(("commit",))

        @staticmethod
        def _finalize_checkpoint_restart():
            calls.append(("finalize",))

        @staticmethod
        def _rollback_checkpoint_restart():
            calls.append(("rollback",))

    owner = SimpleNamespace(
        _execution_context=SimpleNamespace(
            communicator=SimpleNamespace(identity="serial", handle=None)
        )
    )

    assert (
        restore_checkpoint_payload(
            owner,
            _Executor(),
            b"checkpoint",
            bit_identical=True,
        )
        == "restored"
    )
    assert calls == [
        ("prepare", b"checkpoint", True),
        ("begin",),
        ("apply", "prepared"),
        ("commit",),
        ("finalize",),
    ]


def test_collective_restart_authenticates_regrid_mode_and_identity_before_prepare():
    from pops.output import RegridOnRestart
    from pops.output._checkpoint_collective import restore_checkpoint_payload

    hierarchy = RegridOnRestart()
    calls = []

    class _Executor:
        @staticmethod
        def _prepare_checkpoint_restart(
            payload,
            *,
            bit_identical,
            hierarchy_mode,
            hierarchy_identity,
        ):
            calls.append(
                (
                    "prepare",
                    payload,
                    bit_identical,
                    hierarchy_mode,
                    hierarchy_identity,
                )
            )
            return "prepared"

        @staticmethod
        def _begin_checkpoint_restart():
            calls.append(("begin",))

        @staticmethod
        def _apply_checkpoint_restart(prepared):
            calls.append(("apply", prepared))
            return "restored"

        @staticmethod
        def _commit_checkpoint_restart():
            calls.append(("commit",))

        @staticmethod
        def _finalize_checkpoint_restart():
            calls.append(("finalize",))

        @staticmethod
        def _rollback_checkpoint_restart():
            calls.append(("rollback",))

    owner = SimpleNamespace(
        _execution_context=SimpleNamespace(
            communicator=SimpleNamespace(identity="serial", handle=None)
        )
    )
    assert (
        restore_checkpoint_payload(
            owner,
            _Executor(),
            b"checkpoint",
            bit_identical=False,
            hierarchy_mode="regrid_on_restart",
            hierarchy_identity=hierarchy.identity.token,
        )
        == "restored"
    )
    assert calls == [
        (
            "prepare",
            b"checkpoint",
            False,
            "regrid_on_restart",
            hierarchy.identity.token,
        ),
        ("begin",),
        ("apply", "prepared"),
        ("commit",),
        ("finalize",),
    ]


def test_collective_restart_refuses_an_ignored_recorded_hierarchy_identity():
    from pops.output import RestoreRecordedHierarchy
    from pops.output._checkpoint_collective import restore_checkpoint_payload

    class _UnreachableExecutor:
        def __getattr__(self, name):
            raise AssertionError("restart protocol must not be inspected: %s" % name)

    owner = SimpleNamespace(
        _execution_context=SimpleNamespace(
            communicator=SimpleNamespace(identity="serial", handle=None)
        )
    )
    hierarchy = RestoreRecordedHierarchy()

    with pytest.raises(ValueError, match="only valid with RegridOnRestart"):
        restore_checkpoint_payload(
            owner,
            _UnreachableExecutor(),
            b"checkpoint",
            bit_identical=False,
            hierarchy_mode=hierarchy.mode,
            hierarchy_identity=hierarchy.identity.token,
        )


def test_regrid_restart_derives_distinct_run_identity_from_global_receipt(monkeypatch):
    from pops.output import RegridOnRestart
    from pops.output import _checkpoint_collective
    from pops.runtime._amr_system_io import _AMRRegridRestartEvidence

    source_run_identity = make_identity("run", {"test": "source"})
    restart_identity = make_identity("restart", {"test": "checkpoint"})
    hierarchy = RegridOnRestart()
    receipt = {
        "schema_version": 3,
        "policy_identity": hierarchy.identity.token,
        "changed": True,
        "accepted_time": 0.5,
        "accepted_macro_step": 7,
        "before": {"topology_epoch": 3},
        "after": {"topology_epoch": 4},
        "accepted_contract_identity_before": make_identity(
            "restart-accepted-contract", {"phase": "before"}
        ).token,
        "accepted_contract_identity_after": make_identity(
            "restart-accepted-contract", {"phase": "after"}
        ).token,
        "history_consensus_identity_before": make_identity(
            "restart-history-image", {"phase": "before"}
        ).token,
        "history_consensus_identity_after": make_identity(
            "restart-history-image", {"phase": "after"}
        ).token,
        "field_manifest_identity_before": make_identity(
            "restart-field-provider-manifest", {"phase": "before"}
        ).token,
        "field_manifest_identity_after": make_identity(
            "restart-field-provider-manifest", {"phase": "after"}
        ).token,
        "field_manifest_before": [],
        "field_manifest_after": [],
        "field_recompute_witness": [],
        "composite_integrals_before": [{"block": "tracer", "component": 0, "value": 1.25}],
        "composite_integrals_after": [{"block": "tracer", "component": 0, "value": 1.25}],
    }
    events = []
    published = []

    class _Publisher:
        @staticmethod
        def validate_diagnostic_restart_state(data):
            return data

        @staticmethod
        def restore_diagnostic_restart_state(_data):
            events.append("diagnostics")

        @staticmethod
        def diagnostic_restart_state():
            return {"schema_version": 1}

    class _SnapshotBuilder:
        def __init__(self):
            self._geometry_cache = {}

        @staticmethod
        def invalidate_geometry_cache():
            events.append("geometry")

    class _Native:
        def __init__(self):
            self._last_run_manifest = None
            self._last_run_identity = None
            self._restart_lineage_identity = None

        @staticmethod
        def last_restart_regrid_receipt():
            return receipt

        @staticmethod
        def _restore_checkpoint_run_identity(identity):
            published.append(identity)
            events.append("run")

    def restore_checkpoint_payload(
        owner,
        executor,
        payload,
        *,
        bit_identical,
        hierarchy_mode,
        hierarchy_identity,
        phase_prefix,
        prepare_outer_state,
        after_native_apply,
        rollback_after_native_apply,
    ):
        assert owner is runtime and executor is native
        assert payload == b"checkpoint" and bit_identical is False
        assert hierarchy_mode == "regrid_on_restart"
        assert hierarchy_identity == hierarchy.identity.token
        assert phase_prefix == "native restart"
        assert callable(prepare_outer_state)
        assert callable(after_native_apply)
        assert callable(rollback_after_native_apply)
        prepare_outer_state()
        events.append("native")
        result = _AMRRegridRestartEvidence(restart_identity, receipt)
        after_native_apply(result)
        return result

    native = _Native()
    from pops.runtime._checkpoint_resource_budget import _producer_checkpoint_resource_budget

    resource_budget = _producer_checkpoint_resource_budget(
        {"checkpoint": np.frombuffer(b"checkpoint", dtype=np.uint8)},
        runtime_kind="amr",
        authority="test-amr-regrid-restore",
    )
    runtime = SimpleNamespace(
        _executor=native,
        _snapshot_builder=_SnapshotBuilder(),
        _publisher=_Publisher(),
        _consumer_cursors=None,
        _checkpoint_resource_budget=resource_budget,
    )
    monkeypatch.setattr(
        _checkpoint_collective,
        "decode_checkpoint_bytes",
        lambda _payload, _budget: {
            "runtime_consumer_diagnostics": np.array(json.dumps({"schema_version": 1}))
        },
    )
    monkeypatch.setattr(
        _checkpoint_collective,
        "restore_checkpoint_payload",
        restore_checkpoint_payload,
    )
    monkeypatch.setattr(
        "pops.runtime._checkpoint_manifest.checkpoint_run_identity",
        lambda _payload: source_run_identity,
    )

    assert (
        RuntimeInstance._restore_checkpoint(
            runtime,
            b"checkpoint",
            ConsumerCursorSet(),
            bit_identical=False,
            hierarchy_mode="regrid_on_restart",
            hierarchy_identity=hierarchy.identity.token,
        )
        == restart_identity
    )
    assert len(published) == 1
    assert published[0].domain == "run"
    assert published[0] != source_run_identity
    assert events == ["native", "geometry", "diagnostics", "run"]
    receipt_identity = {
        **receipt,
        "accepted_time": receipt["accepted_time"].hex(),
        "composite_integrals_before": [
            {**row, "value": row["value"].hex()} for row in receipt["composite_integrals_before"]
        ],
        "composite_integrals_after": [
            {**row, "value": row["value"].hex()} for row in receipt["composite_integrals_after"]
        ],
    }
    expected = make_identity(
        "run",
        {
            "continuation": "regrid_on_restart",
            "source_run_identity": source_run_identity.to_data(),
            "restart_identity": restart_identity.to_data(),
            "hierarchy_policy_identity": hierarchy.identity.to_data(),
            "regrid_receipt": receipt_identity,
        },
    )
    assert published[0] == expected


def test_checkpoint_restart_authenticates_and_restores_consumer_cursors(tmp_path):
    plan, _, manifest = _with_graph(tmp_path / "outputs")
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._run(t_end=1.0, max_steps=1)
    checkpoint = runtime.checkpoint(tmp_path / "restart")

    restored = RuntimeInstance(plan, executor=_Executor(plan))
    restored.restart(checkpoint)

    assert restored.last_run_identity == runtime.last_run_identity
    assert restored.consumer_cursors.for_consumer(
        manifest.qualified_id
    ) == runtime.consumer_cursors.for_consumer(manifest.qualified_id)
    assert restored.time() == runtime.time()
    restarted_checkpoint = restored.checkpoint(tmp_path / "restart-after-restart")
    assert Path(restarted_checkpoint).is_file()
    with np.load(checkpoint, allow_pickle=False) as payload:
        assert str(payload["runtime_consumer_graph"]) == runtime.consumer_graph.identity.token
        assert "runtime_consumer_cursors" in payload.files
        diagnostic_state = json.loads(str(payload["runtime_consumer_diagnostics"]))
    assert diagnostic_state == {
        "schema_version": 2,
        "baselines": {},
        "diagnostics": [],
    }


def test_checkpoint_diagnostic_baseline_schema_is_finite_and_canonical():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    canonical = {
        "schema_version": 2,
        "baselines": {"diagnostic:integral": (1.25).hex()},
        "diagnostics": [],
    }
    assert RuntimeConsumerPublisher.validate_diagnostic_restart_state(canonical) == canonical
    with pytest.raises(ValueError, match="finite"):
        RuntimeConsumerPublisher.validate_diagnostic_restart_state(
            {
                "schema_version": 2,
                "baselines": {"diagnostic:integral": "nan"},
                "diagnostics": [],
            }
        )
    with pytest.raises(ValueError, match="canonical"):
        RuntimeConsumerPublisher.validate_diagnostic_restart_state(
            {
                "schema_version": 2,
                "baselines": {"diagnostic:integral": "0x1.4p+0"},
                "diagnostics": [],
            }
        )


def test_root_output_lane_requires_one_active_run_scoped_communicator():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    publisher = object.__new__(RuntimeConsumerPublisher)
    lane = SimpleNamespace(active=True, closed=False)
    publisher._root_output_consumers = ("scientific_output/root",)
    publisher._root_output_lanes = {"run": lane}
    assert publisher._root_output_communicator() is lane

    publisher._root_output_lanes = {}
    with pytest.raises(RuntimeError, match="exactly one active"):
        publisher._root_output_communicator()

    publisher._root_output_lanes = {"run": SimpleNamespace(active=False, closed=False)}
    with pytest.raises(RuntimeError, match="not active"):
        publisher._root_output_communicator()

    publisher._root_output_consumers = ()
    with pytest.raises(RuntimeError, match="declares no ROOT"):
        publisher._root_output_communicator()


def test_root_output_lane_is_materialized_and_closed_once_per_run():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    class _Lane:
        active = True
        closed = False

        def __init__(self):
            self.close_calls = 0
            self.identity = ""

        def close_collectively(self):
            self.close_calls += 1
            self.active = False
            self.closed = True

    class _World:
        identity = "MPI_COMM_WORLD"

        def __init__(self, lane):
            self.lane = lane
            self.identities = []

        def duplicate_observer_lane(self, identity):
            self.identities.append(identity)
            self.lane.identity = "%s/%s" % (self.identity, identity)
            return self.lane

    run_identity = make_identity("run", {"case": "root-output-lane"})
    lane = _Lane()
    world = _World(lane)
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 1
    publisher._root_output_consumers = ("scientific_output/root",)
    publisher._root_output_lanes = {}
    publisher._communicator = world
    publisher._closed_observer_runs = set()
    publisher._observer_run_phases = {}
    publisher._builtin_catalyst_consumers = ()
    publisher._builtin_catalyst_run_started = False
    publisher._owner = SimpleNamespace(
        _consumer_graph=SimpleNamespace(nodes=()),
    )
    publisher._observer_diagnostics = []
    publisher._observer_workers = {}
    publisher._observer_reports = {}
    publisher._observer_queues = {}
    publisher._observer_lanes = {}
    publisher._observer_pending_failures = {}

    publisher.begin_post_commit_consumers(run_identity)
    assert world.identities == ["scientific-output/root/%s" % run_identity.token]
    assert publisher._root_output_communicator() is lane

    assert publisher.close_live_visualizations(run_identity) == ()
    assert lane.close_calls == 1
    assert publisher.close_live_visualizations(run_identity) == ()
    assert lane.close_calls == 1
    with pytest.raises(RuntimeError, match="already closed"):
        publisher.begin_post_commit_consumers(run_identity)


def test_root_lane_close_retains_cleanup_authority_for_retry():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    class _Lane:
        active = True
        closed = False
        fail = True
        identity = ""

        def close_collectively(self):
            if self.fail:
                raise RuntimeError("injected collective close failure")
            self.active = False
            self.closed = True

    class _World:
        identity = "MPI_COMM_WORLD"

        def duplicate_observer_lane(self, identity):
            lane.identity = "%s/%s" % (self.identity, identity)
            return lane

    run_identity = make_identity("run", {"case": "retained-root-lane-cleanup"})
    lane = _Lane()
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 1
    publisher._root_output_consumers = ("scientific_output/root",)
    publisher._root_output_lanes = {}
    publisher._communicator = _World()
    publisher._closed_observer_runs = set()
    publisher._observer_run_phases = {}
    publisher._builtin_catalyst_consumers = ()
    publisher._builtin_catalyst_run_started = False
    publisher._owner = SimpleNamespace(_consumer_graph=SimpleNamespace(nodes=()))
    publisher._observer_diagnostics = []
    publisher._observer_workers = {}
    publisher._observer_reports = {}
    publisher._observer_queues = {}
    publisher._observer_lanes = {}
    publisher._observer_journals = {}
    publisher._observer_pending_failures = {}

    publisher.begin_post_commit_consumers(run_identity)
    with pytest.raises(RuntimeError, match="injected collective close failure"):
        publisher.close_live_visualizations(run_identity)
    assert publisher._root_output_lanes[run_identity.token] is lane
    assert run_identity.token in publisher._closed_observer_runs

    lane.fail = False
    publisher.close_live_visualizations(run_identity)
    assert run_identity.token not in publisher._root_output_lanes
    assert lane.closed is True
    assert run_identity.token in publisher._closed_observer_runs


@pytest.mark.parametrize(
    ("peer_error", "peer_present", "sentinel_retained"),
    (
        ("injected peer ROOT lane construction failure", False, False),
        (None, True, True),
    ),
    ids=("all-ranks-fail", "mixed-rank-success"),
)
def test_root_lane_construction_failure_reaches_world_consensus_before_exit(
    monkeypatch,
    peer_error,
    peer_present,
    sentinel_retained,
):
    from pops.runtime import _runtime_consumers
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    class _World:
        identity = "MPI_COMM_WORLD"

        def __init__(self):
            self.duplicate_calls = 0

        def duplicate_observer_lane(self, _identity):
            self.duplicate_calls += 1
            raise RuntimeError("injected local ROOT lane construction failure")

    run_identity = make_identity("run", {"case": "root-lane-construction-consensus"})
    world = _World()
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 2
    publisher._communicator = world
    publisher._root_output_consumers = ("scientific_output/root",)
    publisher._root_output_lanes = {}
    publisher._closed_observer_runs = set()
    publisher._observer_run_phases = {}
    publisher._builtin_catalyst_consumers = ()
    publisher._builtin_catalyst_run_started = False
    publisher._owner = SimpleNamespace(_consumer_graph=SimpleNamespace(nodes=()))

    consensus_envelopes = []

    def gathered(_communicator, envelope):
        consensus_envelopes.append(dict(envelope))
        peer = dict(envelope)
        peer["rank"] = 1
        peer["error"] = peer_error
        peer["present"] = peer_present
        return envelope, peer

    monkeypatch.setattr(_runtime_consumers, "allgather_value", gathered)

    with pytest.raises(
        _runtime_consumers._ObserverCollectiveRejected,
        match="ROOT scientific-output lane construction failed",
    ):
        publisher.begin_post_commit_consumers(run_identity)

    assert world.duplicate_calls == 1
    assert len(consensus_envelopes) == 1
    assert "injected local ROOT lane construction failure" in consensus_envelopes[0]["error"]
    assert (run_identity.token in publisher._root_output_lanes) is sentinel_retained
    if sentinel_retained:
        assert publisher._root_output_lanes[run_identity.token] is None
    assert publisher._observer_run_phases[run_identity.token] == "opening"


def test_only_consumer_free_serial_failed_run_releases_its_identity_for_retry():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    class _Lane:
        active = True
        closed = False
        identity = ""

        def close_collectively(self):
            self.active = False
            self.closed = True

    class _World:
        identity = "MPI_COMM_WORLD"

        def __init__(self):
            self.lanes = []

        def duplicate_observer_lane(self, identity):
            lane = _Lane()
            lane.identity = "%s/%s" % (self.identity, identity)
            self.lanes.append(lane)
            return lane

    run_identity = make_identity("run", {"case": "retryable-consumer-free-serial"})
    world = _World()
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 1
    publisher._root_output_consumers = ()
    publisher._root_output_lanes = {}
    publisher._communicator = None
    publisher._closed_observer_runs = set()
    publisher._observer_run_phases = {}
    publisher._builtin_catalyst_consumers = ()
    publisher._builtin_catalyst_run_started = False
    publisher._owner = SimpleNamespace(
        _consumer_graph=SimpleNamespace(nodes=()),
    )
    publisher._observer_diagnostics = []
    publisher._observer_workers = {}
    publisher._observer_reports = {}
    publisher._observer_queues = {}
    publisher._observer_lanes = {}
    publisher._observer_journals = {}
    publisher._observer_preflight_sessions = {}
    publisher._observer_pending_failures = {}

    entry_fence = publisher.failed_run_effect_fence()
    publisher.begin_post_commit_consumers(run_identity)
    publisher.close_failed_run_consumers(
        run_identity,
        release_identity=True,
        entry_effect_fence=entry_fence,
    )
    assert run_identity.token not in publisher._closed_observer_runs

    publisher.begin_post_commit_consumers(run_identity)
    publisher.close_live_visualizations(run_identity)
    assert run_identity.token in publisher._closed_observer_runs
    publisher.close_failed_run_consumers(run_identity, release_identity=True)
    assert run_identity.token in publisher._closed_observer_runs

    published_identity = make_identity("run", {"case": "published-at-start"})
    publisher.begin_post_commit_consumers(published_identity)
    publisher.close_failed_run_consumers(
        published_identity,
        release_identity=False,
    )
    assert published_identity.token in publisher._closed_observer_runs

    output_identity = make_identity("run", {"case": "root-output-opened"})
    publisher._root_output_consumers = ("scientific_output/root",)
    publisher._communicator = world
    output_fence = publisher.failed_run_effect_fence()
    publisher.begin_post_commit_consumers(output_identity)
    publisher.close_failed_run_consumers(
        output_identity,
        release_identity=True,
        entry_effect_fence=output_fence,
    )
    assert output_identity.token in publisher._closed_observer_runs
    assert world.lanes[0].closed is True
    publisher._root_output_consumers = ()
    publisher._communicator = None

    diagnostic_identity = make_identity("run", {"case": "diagnostic-before-failure"})
    diagnostic_fence = publisher.failed_run_effect_fence()
    publisher.begin_post_commit_consumers(diagnostic_identity)
    publisher._observer_diagnostics.append("provider initialization escaped rollback")
    publisher.close_failed_run_consumers(
        diagnostic_identity,
        release_identity=True,
        entry_effect_fence=diagnostic_fence,
    )
    assert diagnostic_identity.token in publisher._closed_observer_runs

    catalyst_identity = make_identity("run", {"case": "catalyst-begin-failure"})
    publisher._builtin_catalyst_consumers = ("monitor/catalyst",)
    catalyst_fence = publisher.failed_run_effect_fence()
    publisher.begin_post_commit_consumers(catalyst_identity)
    publisher.close_failed_run_consumers(
        catalyst_identity,
        release_identity=True,
        entry_effect_fence=catalyst_fence,
    )
    assert catalyst_identity.token in publisher._closed_observer_runs


def test_mpi_size_one_consumer_free_failed_run_releases_its_identity():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    run_identity = make_identity("run", {"case": "mpi-size-one-reusable"})
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 1
    publisher._communicator = SimpleNamespace(identity="MPI_COMM_WORLD")
    publisher._root_output_consumers = ()
    publisher._root_output_lanes = {}
    publisher._closed_observer_runs = set()
    publisher._observer_run_phases = {}
    publisher._builtin_catalyst_consumers = ()
    publisher._builtin_catalyst_run_started = False
    publisher._owner = SimpleNamespace(_consumer_graph=SimpleNamespace(nodes=()))
    publisher._observer_diagnostics = []
    publisher._observer_workers = {}
    publisher._observer_reports = {}
    publisher._observer_queues = {}
    publisher._observer_lanes = {}
    publisher._observer_journals = {}
    publisher._observer_preflight_sessions = {}
    publisher._observer_pending_failures = {}

    entry_fence = publisher.failed_run_effect_fence()
    publisher.begin_post_commit_consumers(run_identity)
    publisher.close_failed_run_consumers(
        run_identity,
        release_identity=True,
        entry_effect_fence=entry_fence,
    )
    assert run_identity.token not in publisher._closed_observer_runs
    assert run_identity.token not in publisher._observer_run_phases
    publisher.begin_post_commit_consumers(run_identity)


def test_mpi_multi_rank_consumer_free_failed_run_keeps_its_identity(monkeypatch):
    from pops.runtime import _runtime_consumers
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    run_identity = make_identity("run", {"case": "mpi-multi-rank-sealed"})
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 2
    publisher._communicator = SimpleNamespace(identity="MPI_COMM_WORLD")
    publisher._root_output_consumers = ()
    publisher._root_output_lanes = {}
    publisher._closed_observer_runs = set()
    publisher._observer_run_phases = {}
    publisher._builtin_catalyst_consumers = ()
    publisher._builtin_catalyst_run_started = False
    publisher._owner = SimpleNamespace(_consumer_graph=SimpleNamespace(nodes=()))
    publisher._observer_diagnostics = []
    publisher._observer_workers = {}
    publisher._observer_reports = {}
    publisher._observer_queues = {}
    publisher._observer_lanes = {}
    publisher._observer_journals = {}
    publisher._observer_preflight_sessions = {}
    publisher._observer_pending_failures = {}

    def consensus_rows(_communicator, envelope):
        peer = dict(envelope)
        peer["rank"] = 1
        return envelope, peer

    monkeypatch.setattr(_runtime_consumers, "allgather_value", consensus_rows)
    entry_fence = publisher.failed_run_effect_fence()
    publisher.begin_post_commit_consumers(run_identity)
    publisher.close_failed_run_consumers(
        run_identity,
        release_identity=True,
        entry_effect_fence=entry_fence,
    )
    assert run_identity.token in publisher._closed_observer_runs
    assert publisher._observer_run_phases[run_identity.token] == "closed"


def test_runtime_instance_exposes_only_exact_native_program_accepted_state():
    runtime = object.__new__(RuntimeInstance)
    runtime._executor = SimpleNamespace(program_accepted_state=lambda: b"accepted-amr-state")
    assert runtime.program_accepted_state() == b"accepted-amr-state"

    runtime._executor = SimpleNamespace()
    with pytest.raises(NotImplementedError, match="accepted AMR Program state"):
        runtime.program_accepted_state()

    runtime._executor = SimpleNamespace(program_accepted_state=lambda: bytearray(b"mutable"))
    with pytest.raises(TypeError, match="must be exact bytes"):
        runtime.program_accepted_state()


def test_failed_run_close_refuses_divergent_mpi_lane_inventory(monkeypatch):
    from pops.runtime import _runtime_consumers
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    run_identity = make_identity("run", {"case": "rank-divergent-release"})
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 2
    publisher._communicator = object()
    publisher._root_output_consumers = ()
    publisher._root_output_lanes = {}
    publisher._closed_observer_runs = set()
    publisher._observer_run_phases = {run_identity.token: "opening"}
    publisher._builtin_catalyst_consumers = ()
    publisher._builtin_catalyst_run_started = False
    manifest = SimpleNamespace(
        kind=ConsumerKind.MONITOR,
        qualified_id="monitor/collective",
        parallel_mode=ParallelMode.COLLECTIVE,
        identity=make_identity("consumer-manifest", {"case": "collective-close"}),
        operation_data={"observer": {"provider": {"provider_id": "test.collective-observer"}}},
    )
    publisher._owner = SimpleNamespace(_consumer_graph=SimpleNamespace(nodes=(manifest,)))
    publisher._observer_diagnostics = []
    publisher._observer_workers = {}
    publisher._observer_reports = {}
    publisher._observer_queues = {}
    publisher._observer_lanes = {}
    publisher._observer_journals = {}
    publisher._observer_pending_failures = {}

    entry_fence = publisher.failed_run_effect_fence()

    def divergent_rows(_communicator, envelope):
        peer = dict(envelope)
        peer["rank"] = 1
        peer["monitors"] = [dict(row) for row in envelope["monitors"]]
        peer["monitors"][0]["lane"] = {
            "identity": "MPI_COMM_WORLD/post-commit/%s/%s"
            % (manifest.identity.token, run_identity.token),
            "active": True,
            "closed": False,
        }
        return envelope, peer

    monkeypatch.setattr(_runtime_consumers, "allgather_value", divergent_rows)
    with pytest.raises(RuntimeError, match="divergent MPI monitor inventory"):
        publisher.close_failed_run_consumers(
            run_identity,
            release_identity=True,
            entry_effect_fence=entry_fence,
        )
    assert run_identity.token in publisher._closed_observer_runs


def test_opened_root_output_refuses_close_after_lane_authority_disappears():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    run_identity = make_identity("run", {"case": "missing-open-root-lane"})
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 1
    publisher._communicator = object()
    publisher._root_output_consumers = ("scientific_output/root",)
    publisher._root_output_lanes = {}
    publisher._closed_observer_runs = set()
    publisher._observer_run_phases = {run_identity.token: "open"}
    publisher._owner = SimpleNamespace(_consumer_graph=SimpleNamespace(nodes=()))
    publisher._observer_workers = {}
    publisher._observer_queues = {}
    publisher._observer_lanes = {}
    publisher._observer_reports = {}
    publisher._observer_pending_failures = {}
    publisher._observer_diagnostics = []

    with pytest.raises(RuntimeError, match="lost its opened ROOT output lane"):
        publisher.close_live_visualizations(run_identity)
    assert run_identity.token in publisher._closed_observer_runs


def test_close_preflight_refuses_queue_owned_by_another_run():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    run_identity = make_identity("run", {"case": "queue-owner"})
    other_identity = make_identity("run", {"case": "other-queue-owner"})
    manifest = SimpleNamespace(
        kind=ConsumerKind.MONITOR,
        qualified_id="monitor/serial",
        parallel_mode=ParallelMode.SERIAL,
        operation_data={"observer": {"provider": {"provider_id": "test.serial-observer"}}},
    )
    queue = SimpleNamespace(
        close_authority={
            "run_identity": other_identity.token,
            "consumer_id": manifest.qualified_id,
            "provider_id": "test.serial-observer",
        },
        close_requested=False,
        close_succeeded=False,
    )
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 1
    publisher._communicator = None
    publisher._root_output_consumers = ()
    publisher._root_output_lanes = {}
    publisher._observer_run_phases = {run_identity.token: "open"}
    publisher._owner = SimpleNamespace(_consumer_graph=SimpleNamespace(nodes=(manifest,)))
    publisher._observer_workers = {}
    publisher._observer_queues = {(manifest.qualified_id, run_identity.token): queue}
    publisher._observer_lanes = {}

    with pytest.raises(RuntimeError, match="queue owned by another run or consumer"):
        publisher._preflight_observer_close(run_identity)


def test_pending_observer_abort_retries_only_after_local_failure():
    from pops.output.observers import authenticate_observer_session
    from pops.runtime._runtime_consumers import _PendingObserverSession

    class _Session:
        authority = {
            "schema_version": 1,
            "provider_id": "test.pending-observer",
            "delivery": "post_commit",
            "threading": "dedicated_serial",
            "worker_mpi": False,
        }

        def __init__(self):
            self.abort_calls = 0

        def initialize(self, _run):
            return None

        def execute(self, _frame):
            raise AssertionError("unused")

        def finalize(self):
            return None

        def abort(self):
            self.abort_calls += 1
            if self.abort_calls == 1:
                raise RuntimeError("transient abort failure")

    run_identity = make_identity("run", {"case": "pending-abort-retry"})
    session = _Session()
    pending = _PendingObserverSession(
        run_identity,
        "monitor/pending",
        session.authority["provider_id"],
        False,
        session,
    )
    assert authenticate_observer_session(pending)["provider_id"] == "test.pending-observer"

    with pytest.raises(RuntimeError, match="transient abort failure"):
        pending.abort()
    assert not pending.abort_succeeded

    pending.abort()
    pending.abort()
    assert pending.abort_succeeded
    assert session.abort_calls == 2


def test_pending_observer_abort_marks_worker_collective_loss():
    from pops.output.observers import ObserverWorkerCollectiveLost
    from pops.runtime._runtime_consumers import _PendingObserverSession

    class _Session:
        authority = {
            "schema_version": 1,
            "provider_id": "test.pending-observer-lost-lane",
            "delivery": "post_commit",
            "threading": "dedicated_collective",
            "worker_mpi": True,
        }

        def initialize(self, _run):
            return None

        def execute(self, _frame):
            raise AssertionError("unused")

        def finalize(self):
            return None

        def abort(self):
            raise ObserverWorkerCollectiveLost("injected pending abort lane loss")

    run_identity = make_identity("run", {"case": "pending-abort-lost-lane"})
    session = _Session()
    pending = _PendingObserverSession(
        run_identity,
        "monitor/pending-lost-lane",
        session.authority["provider_id"],
        True,
        session,
    )

    with pytest.raises(ObserverWorkerCollectiveLost, match="pending abort lane loss"):
        pending.abort()
    assert pending.abort_succeeded is False
    assert pending.worker_collective_lost is True


def test_failed_pending_session_retains_worker_until_owner_thread_retry():
    from pops.runtime._observer_runtime import PostCommitObserverWorker
    from pops.runtime._runtime_consumers import (
        _PendingObserverSession,
        RuntimeConsumerPublisher,
    )

    class _Session:
        authority = {
            "schema_version": 1,
            "provider_id": "test.pending-worker-owner",
            "delivery": "post_commit",
            "threading": "dedicated_serial",
            "worker_mpi": False,
        }

        def __init__(self):
            self.abort_threads = []

        def initialize(self, _run):
            return None

        def execute(self, _frame):
            raise AssertionError("unused")

        def finalize(self):
            return None

        def abort(self):
            self.abort_threads.append(threading.get_ident())
            if len(self.abort_threads) == 1:
                raise RuntimeError("transient owner-thread abort failure")

    run_identity = make_identity("run", {"case": "pending-worker-owner-retry"})
    manifest = SimpleNamespace(
        kind=ConsumerKind.MONITOR,
        qualified_id="monitor/pending-worker-owner",
        parallel_mode=ParallelMode.SERIAL,
        identity=make_identity("consumer-manifest", {"case": "pending-worker-owner"}),
        operation_data={
            "observer": {"provider": {"provider_id": _Session.authority["provider_id"]}},
            "on_failure": {"action": "raise_on_flush"},
        },
    )
    key = (manifest.qualified_id, run_identity.token)
    session = _Session()
    pending = _PendingObserverSession(
        run_identity,
        manifest.qualified_id,
        session.authority["provider_id"],
        False,
        session,
    )
    worker = PostCommitObserverWorker(
        thread_name="test-pending-worker-owner",
        run_identity=run_identity,
    )
    try:
        owner_thread = worker._thread.ident
        assert owner_thread is not None

        publisher = object.__new__(RuntimeConsumerPublisher)
        publisher._rank = 0
        publisher._size = 1
        publisher._communicator = None
        publisher._root_output_consumers = ()
        publisher._root_output_lanes = {}
        publisher._closed_observer_runs = set()
        publisher._observer_run_phases = {run_identity.token: "opening"}
        publisher._owner = SimpleNamespace(_consumer_graph=SimpleNamespace(nodes=(manifest,)))
        publisher._observer_workers = {run_identity.token: worker}
        publisher._observer_pending_sessions = {key: pending}
        publisher._observer_queues = {}
        publisher._observer_lanes = {}
        publisher._observer_pending_failures = {}
        publisher._observer_reports = {}
        publisher._observer_diagnostics = []

        with pytest.raises(RuntimeError, match="transient owner-thread abort failure"):
            publisher.close_live_visualizations(run_identity)
        assert publisher._observer_pending_sessions[key] is pending
        assert publisher._observer_workers[run_identity.token] is worker
        assert worker._thread.is_alive()
        assert worker.close_requested is False
        assert session.abort_threads == [owner_thread]

        assert publisher.close_live_visualizations(run_identity) == ()
        assert session.abort_threads == [owner_thread, owner_thread]
        assert key not in publisher._observer_pending_sessions
        assert run_identity.token not in publisher._observer_workers
        assert worker.close_succeeded is True
        assert worker._thread.is_alive() is False
        assert publisher._observer_run_phases[run_identity.token] == "closed"
    finally:
        if not worker.close_succeeded:
            worker.close()


def test_close_preflight_refuses_divergent_mpi_run_lifecycle_phases(monkeypatch):
    from pops.runtime import _runtime_consumers
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    run_identity = make_identity("run", {"case": "divergent-close-phases"})
    manifest = SimpleNamespace(
        kind=ConsumerKind.MONITOR,
        qualified_id="monitor/divergent-close-phases",
        parallel_mode=ParallelMode.COLLECTIVE,
        identity=make_identity("consumer-manifest", {"case": "divergent-close-phases"}),
        operation_data={
            "observer": {"provider": {"provider_id": "test.collective-observer"}},
            "on_failure": {"action": "raise_on_flush"},
        },
    )
    key = (manifest.qualified_id, run_identity.token)
    authority = {
        "run_identity": run_identity.token,
        "consumer_id": manifest.qualified_id,
        "provider_id": "test.collective-observer",
    }
    queue = SimpleNamespace(
        close_authority=authority,
        close_requested=True,
        close_succeeded=False,
    )
    lane = SimpleNamespace(
        identity="MPI_COMM_WORLD/post-commit/%s/%s" % (manifest.identity.token, run_identity.token),
        active=True,
        closed=False,
    )
    worker = SimpleNamespace(
        close_authority=run_identity.token,
        close_requested=False,
        close_succeeded=False,
    )
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 2
    publisher._communicator = SimpleNamespace(identity="MPI_COMM_WORLD")
    publisher._root_output_consumers = ()
    publisher._root_output_lanes = {}
    publisher._closed_observer_runs = set()
    publisher._observer_run_phases = {run_identity.token: "closing_opening"}
    publisher._owner = SimpleNamespace(_consumer_graph=SimpleNamespace(nodes=(manifest,)))
    publisher._observer_workers = {run_identity.token: worker}
    publisher._observer_pending_sessions = {}
    publisher._observer_queues = {key: queue}
    publisher._observer_lanes = {key: lane}

    def divergent_phase_rows(_communicator, envelope):
        assert envelope["phase"] == "closing_opening"
        peer = dict(envelope)
        peer["rank"] = 1
        peer["phase"] = "closing_open"
        assert all(
            peer[name] == envelope[name] for name in ("error", "root_lane", "worker", "monitors")
        )
        return envelope, peer

    monkeypatch.setattr(_runtime_consumers, "allgather_value", divergent_phase_rows)
    drain_calls = []

    def forbidden_drain(*_args, **_kwargs):
        drain_calls.append(True)
        raise AssertionError("divergent phases must be refused before abort or finalize")

    publisher._drain_observer_manifest = forbidden_drain

    with pytest.raises(RuntimeError, match="divergent run lifecycle phases"):
        publisher.close_live_visualizations(run_identity)

    assert drain_calls == []
    assert publisher._observer_queues[key] is queue
    assert publisher._observer_lanes[key] is lane
    assert publisher._observer_workers[run_identity.token] is worker


def test_opening_preflight_refuses_complementary_mpi_owners_with_partial_worker(monkeypatch):
    from pops.runtime import _runtime_consumers
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    run_identity = make_identity("run", {"case": "complementary-opening-owners"})
    manifest = SimpleNamespace(
        kind=ConsumerKind.MONITOR,
        qualified_id="monitor/collective-opening",
        parallel_mode=ParallelMode.COLLECTIVE,
        identity=make_identity("consumer-manifest", {"case": "collective-opening"}),
        operation_data={"observer": {"provider": {"provider_id": "test.collective-observer"}}},
    )
    key = (manifest.qualified_id, run_identity.token)
    authority = {
        "run_identity": run_identity.token,
        "consumer_id": manifest.qualified_id,
        "provider_id": "test.collective-observer",
    }
    queue = SimpleNamespace(
        close_authority=authority,
        close_requested=False,
        close_succeeded=False,
    )
    lane_identity = "MPI_COMM_WORLD/post-commit/%s/%s" % (
        manifest.identity.token,
        run_identity.token,
    )
    lane = SimpleNamespace(identity=lane_identity, active=True, closed=False)
    worker = SimpleNamespace(
        close_authority=run_identity.token,
        close_requested=False,
        close_succeeded=False,
    )
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 2
    publisher._communicator = SimpleNamespace(identity="MPI_COMM_WORLD")
    publisher._root_output_consumers = ()
    publisher._root_output_lanes = {}
    publisher._observer_run_phases = {run_identity.token: "opening"}
    publisher._owner = SimpleNamespace(_consumer_graph=SimpleNamespace(nodes=(manifest,)))
    publisher._observer_workers = {run_identity.token: worker}
    publisher._observer_pending_sessions = {}
    publisher._observer_queues = {key: queue}
    publisher._observer_lanes = {key: lane}

    def complementary_peer(_communicator, envelope):
        peer = dict(envelope)
        peer["rank"] = 1
        peer["worker"] = None
        peer["monitors"] = [dict(row) for row in envelope["monitors"]]
        peer["monitors"][0]["session"] = {
            "authority": authority,
            "abort_succeeded": False,
            "authenticated": True,
        }
        peer["monitors"][0]["queue"] = None
        return envelope, peer

    monkeypatch.setattr(_runtime_consumers, "allgather_value", complementary_peer)
    with pytest.raises(RuntimeError, match="without every run worker"):
        publisher._preflight_observer_close(run_identity)


def test_rank_divergent_collective_abort_is_retained_without_unsafe_retry(monkeypatch):
    from pops.runtime import _runtime_consumers
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    run_identity = make_identity("run", {"case": "partial-collective-abort"})
    manifest = SimpleNamespace(
        kind=ConsumerKind.MONITOR,
        qualified_id="monitor/partial-abort",
        parallel_mode=ParallelMode.COLLECTIVE,
        operation_data={"on_failure": {"action": "raise_on_flush"}},
    )
    key = (manifest.qualified_id, run_identity.token)

    class _Queue:
        close_requested = False
        close_succeeded = False
        reports = ()

        def __init__(self):
            self.abort_calls = 0
            self.prepare_calls = 0

        def prepare_abort_close(self):
            self.prepare_calls += 1
            self.close_requested = True
            return ()

        def prepare_complete_abort_close(self):
            return None

        def cancel_complete_abort_close(self, _error):
            return None

        def arm_complete_abort_close(self):
            return None

        def complete_abort_close(self):
            self.abort_calls += 1
            self.close_succeeded = True
            return ()

        def close(self):
            raise AssertionError("failed opening must not finalize its observer queue")

    queue = _Queue()
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 2
    publisher._communicator = SimpleNamespace(identity="MPI_COMM_WORLD")
    publisher._observer_run_phases = {run_identity.token: "closing_opening"}
    publisher._observer_pending_sessions = {}
    publisher._observer_queues = {key: queue}
    publisher._observer_lanes = {key: SimpleNamespace(closed=False)}
    publisher._observer_pending_failures = {}
    publisher._observer_abort_retry_blocked = set()
    publisher._observer_reports = {}
    publisher._observer_diagnostics = []

    abort_phases = 0

    def peer_abort_failure(_communicator, envelope):
        nonlocal abort_phases
        peer = dict(envelope)
        peer["rank"] = 1
        if set(envelope) == {"rank", "lost"}:
            peer["lost"] = False
        elif set(envelope) in (
            {"rank", "owned", "ready"},
            {"rank", "owned", "ready", "worker_lane_lost"},
        ):
            abort_phases += 1
            peer["owned"] = True
            peer["ready"] = abort_phases == 1
            if "worker_lane_lost" in envelope:
                peer["worker_lane_lost"] = False
        elif set(envelope) == {"rank", "owned", "error"}:
            peer["owned"] = True
            peer["error"] = None
        elif set(envelope) == {"rank", "ready"}:
            peer["ready"] = False
        elif set(envelope) == {"rank", "reports", "diagnostics"}:
            peer["reports"] = []
            peer["diagnostics"] = []
        else:  # pragma: no cover - every collective phase is authenticated above
            raise AssertionError("unexpected close collective")
        return envelope, peer

    monkeypatch.setattr(_runtime_consumers, "allgather_value", peer_abort_failure)

    first = publisher._drain_observer_manifest(manifest, run_identity, close=True)
    assert any("subset of MPI ranks" in failure for failure in first)
    assert queue.prepare_calls == 1
    assert queue.abort_calls == 1
    assert key in publisher._observer_queues
    assert key in publisher._observer_abort_retry_blocked

    second = publisher._drain_observer_manifest(manifest, run_identity, close=True)
    assert any("retry refused" in failure for failure in second)
    assert queue.prepare_calls == 1
    assert queue.abort_calls == 1
    assert key in publisher._observer_queues


def test_poisoned_worker_lane_refuses_provider_and_lane_cleanup(monkeypatch):
    from pops.runtime import _runtime_consumers
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    run_identity = make_identity("run", {"case": "poisoned-worker-lane-close"})
    manifest = SimpleNamespace(
        kind=ConsumerKind.MONITOR,
        qualified_id="monitor/poisoned-worker-lane",
        parallel_mode=ParallelMode.COLLECTIVE,
        operation_data={"on_failure": {"action": "raise_on_flush"}},
    )
    key = (manifest.qualified_id, run_identity.token)

    class _Queue:
        worker_collective_lost = True
        close_requested = False
        close_succeeded = False
        reports = ()

        def __init__(self):
            self.seal_calls = 0

        def seal_local(self, _error):
            self.seal_calls += 1

        def __getattr__(self, name):
            if name.startswith(("prepare", "arm", "complete", "close", "abort", "flush")):
                raise AssertionError("poisoned worker lane must not reenter queue lifecycle")
            raise AttributeError(name)

    class _Lane:
        def __init__(self):
            self.closed = False
            self.close_calls = 0

        def close_collectively(self):
            self.close_calls += 1
            raise AssertionError("poisoned worker lane must not be reused or closed")

    class _Worker:
        def __init__(self):
            self.seal_calls = 0
            self.close_succeeded = False
            self.stopped = False

        def seal_local(self, _error):
            self.seal_calls += 1
            self.close_succeeded = True
            self.stopped = True

    queue = _Queue()
    lane = _Lane()
    worker = _Worker()
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 2
    publisher._communicator = SimpleNamespace(identity="MPI_COMM_WORLD")
    publisher._observer_run_phases = {run_identity.token: "closing_open"}
    publisher._observer_pending_sessions = {}
    publisher._observer_queues = {key: queue}
    publisher._observer_lanes = {key: lane}
    publisher._observer_workers = {run_identity.token: worker}
    publisher._observer_pending_failures = {}
    publisher._observer_abort_retry_blocked = set()
    publisher._observer_finalize_retry_blocked = set()
    publisher._observer_reports = {}
    publisher._observer_diagnostics = []

    collective_phases = []

    def gathered(_communicator, envelope):
        peer = dict(envelope)
        peer["rank"] = 1
        if set(envelope) == {"rank", "lost"}:
            collective_phases.append("health")
            peer["lost"] = True
        elif set(envelope) == {"rank", "error", "closed"}:
            collective_phases.append("local-worker-seal")
            assert worker.stopped is True
            assert envelope["error"] is None
            assert envelope["closed"] is True
        else:  # pragma: no cover - every worker-loss phase is authenticated above
            raise AssertionError("unexpected poisoned worker collective envelope")
        return envelope, peer

    monkeypatch.setattr(_runtime_consumers, "allgather_value", gathered)

    with pytest.raises(
        _runtime_consumers._ObserverWorkerLaneLost,
        match="worker lane lost collective proof",
    ):
        publisher._drain_observer_manifest(manifest, run_identity, close=True)

    assert collective_phases == ["health", "local-worker-seal"]
    assert queue.seal_calls == 1
    assert worker.seal_calls == 1
    assert worker.close_succeeded is True
    assert worker.stopped is True
    assert publisher._observer_workers[run_identity.token] is worker
    assert publisher._observer_queues[key] is queue
    assert publisher._observer_lanes[key] is lane
    assert lane.close_calls == 0
    assert "worker lane lost collective proof" in publisher._observer_pending_failures[key][0]


def test_pending_abort_worker_lane_loss_stops_worker_without_lane_reentry(monkeypatch, request):
    from pops.output.observers import ObserverWorkerCollectiveLost
    from pops.runtime import _runtime_consumers
    from pops.runtime._observer_runtime import PostCommitObserverWorker
    from pops.runtime._runtime_consumers import (
        _PendingObserverSession,
        RuntimeConsumerPublisher,
    )

    class _Session:
        authority = {
            "schema_version": 1,
            "provider_id": "test.pending-abort-lost-worker-lane",
            "delivery": "post_commit",
            "threading": "dedicated_collective",
            "worker_mpi": True,
        }

        def __init__(self):
            self.abort_calls = 0

        def initialize(self, _run):
            return None

        def execute(self, _frame):
            raise AssertionError("unused")

        def finalize(self):
            return None

        def abort(self):
            self.abort_calls += 1
            raise ObserverWorkerCollectiveLost("injected pending abort worker-lane loss")

    class _Lane:
        closed = False

        def close_collectively(self):
            raise AssertionError("a poisoned worker lane must remain retained")

    run_identity = make_identity("run", {"case": "pending-abort-worker-lane-loss"})
    manifest = SimpleNamespace(
        kind=ConsumerKind.MONITOR,
        qualified_id="monitor/pending-abort-worker-lane-loss",
        parallel_mode=ParallelMode.COLLECTIVE,
        operation_data={"on_failure": {"action": "raise_on_flush"}},
    )
    key = (manifest.qualified_id, run_identity.token)
    session = _Session()
    pending = _PendingObserverSession(
        run_identity,
        manifest.qualified_id,
        session.authority["provider_id"],
        True,
        session,
    )
    worker = PostCommitObserverWorker(
        thread_name="test-pending-abort-worker-lane-loss",
        run_identity=run_identity,
    )
    request.addfinalizer(
        lambda: None if worker.close_succeeded else worker.seal_local(RuntimeError("test cleanup"))
    )
    lane = _Lane()
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 2
    publisher._communicator = SimpleNamespace(identity="MPI_COMM_WORLD")
    publisher._observer_run_phases = {run_identity.token: "closing_opening"}
    publisher._observer_pending_sessions = {key: pending}
    publisher._observer_queues = {}
    publisher._observer_lanes = {key: lane}
    publisher._observer_workers = {run_identity.token: worker}
    publisher._observer_pending_failures = {}
    publisher._observer_abort_retry_blocked = set()
    publisher._observer_finalize_retry_blocked = set()
    publisher._observer_reports = {}
    publisher._observer_diagnostics = []

    phases = []

    def gathered(_communicator, envelope):
        peer = dict(envelope)
        peer["rank"] = 1
        keys = set(envelope)
        if keys == {"rank", "lost"}:
            phases.append("initial-health")
            peer["lost"] = False
        elif keys == {"rank", "owned", "ready"}:
            phases.append("abort-preparation")
        elif keys == {"rank", "owned", "error"}:
            phases.append("abort-admission")
        elif keys == {"rank", "owned", "ready", "worker_lane_lost"}:
            phases.append("abort-completion")
            assert envelope["worker_lane_lost"] is True
            peer["worker_lane_lost"] = True
        elif keys == {"rank", "error", "closed"}:
            phases.append("local-worker-seal")
            assert worker.close_succeeded is True
        else:  # pragma: no cover - every phase is authenticated above
            raise AssertionError("unexpected pending-abort collective envelope")
        return envelope, peer

    monkeypatch.setattr(_runtime_consumers, "allgather_value", gathered)

    with pytest.raises(
        _runtime_consumers._ObserverWorkerLaneLost,
        match="abort lost worker-lane collective proof",
    ):
        publisher._drain_observer_manifest(manifest, run_identity, close=True)

    assert phases == [
        "initial-health",
        "abort-preparation",
        "abort-admission",
        "abort-completion",
        "local-worker-seal",
    ]
    assert session.abort_calls == 1
    assert pending.worker_collective_lost is True
    assert worker.close_succeeded is True
    assert publisher._observer_pending_sessions[key] is pending
    assert publisher._observer_workers[run_identity.token] is worker
    assert publisher._observer_lanes[key] is lane


def test_durable_journal_world_loss_is_not_downgraded_to_local_failure(monkeypatch):
    from pops.runtime import _runtime_consumers
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    manifest = SimpleNamespace(parallel_mode=ParallelMode.COLLECTIVE)
    journal = SimpleNamespace(list_committed=lambda: ())
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 2
    publisher._communicator = SimpleNamespace(identity="MPI_COMM_WORLD")
    collective_calls = 0

    def lost_world(_communicator, _envelope):
        nonlocal collective_calls
        collective_calls += 1
        raise RuntimeError("injected durable WORLD loss")

    monkeypatch.setattr(_runtime_consumers, "allgather_value", lost_world)

    with pytest.raises(
        _runtime_consumers._ObserverCollectiveLost,
        match="lost its WORLD inspection proof",
    ):
        publisher._inspect_observer_journal(manifest, journal)

    assert collective_calls == 1


def test_mpi_finalize_enqueue_consensus_failure_cancels_prepared_provider_call(
    monkeypatch,
    request: pytest.FixtureRequest,
):
    from pops import _native_collectives
    from pops.runtime import _runtime_consumers
    from pops.runtime._observer_runtime import (
        ObserverRun,
        PostCommitObserverQueue,
        PostCommitObserverWorker,
        _PreparedWorkerCall,
    )
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    run_identity = make_identity("run", {"case": "finalize-enqueue-consensus-failure"})
    manifest = SimpleNamespace(
        kind=ConsumerKind.MONITOR,
        qualified_id="monitor/finalize-enqueue-consensus-failure",
        parallel_mode=ParallelMode.COLLECTIVE,
        operation_data={"on_failure": {"action": "raise_on_flush"}},
    )
    key = (manifest.qualified_id, run_identity.token)

    class _Session:
        authority = {
            "schema_version": 1,
            "provider_id": "test.collective-finalize",
            "delivery": "post_commit",
            "threading": "dedicated_collective",
            "worker_mpi": True,
        }

        def __init__(self):
            self.finalize_calls = 0

        def initialize(self, _run):
            return None

        def execute(self, _frame):
            raise AssertionError("finalization admission must not execute an observer frame")

        def finalize(self):
            self.finalize_calls += 1

        def abort(self):
            raise AssertionError("normal finalization must not enter provider abort")

    class _Lane:
        def __init__(self):
            self.closed = False
            self.close_calls = 0

        def close_collectively(self):
            self.close_calls += 1
            self.closed = True

    monkeypatch.setattr(
        _native_collectives,
        "require_communicator",
        lambda communicator, *, allow_world=True: communicator,
    )
    session = _Session()
    lane = _Lane()
    worker = PostCommitObserverWorker(
        thread_name="test-finalize-enqueue-consensus-failure",
        run_identity=run_identity,
    )
    try:
        queue = PostCommitObserverQueue(
            session,
            ObserverRun(run_identity),
            consumer_id=manifest.qualified_id,
            worker_communicator=lane,
            shared_worker=worker,
            defer_initialize=True,
        )
    except BaseException:
        worker.close()
        raise

    def cleanup() -> None:
        cleanup_error = RuntimeError("test cleanup cancelled an unresolved lifecycle call")
        queue.cancel_initialize(cleanup_error)
        queue.cancel_complete_close(cleanup_error)
        queue.cancel_complete_abort_close(cleanup_error)
        worker.close()

    request.addfinalizer(cleanup)

    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 2
    publisher._communicator = SimpleNamespace(identity="MPI_COMM_WORLD")
    publisher._observer_run_phases = {run_identity.token: "open"}
    publisher._observer_pending_sessions = {}
    publisher._observer_queues = {key: queue}
    publisher._observer_lanes = {key: lane}
    publisher._observer_pending_failures = {}
    publisher._observer_abort_retry_blocked = set()
    publisher._observer_finalize_retry_blocked = set()
    publisher._observer_reports = {}
    publisher._observer_diagnostics = []

    prepared_attempts = []
    cancelled_attempts = []
    awaited_attempts = []
    original_cancel = _PreparedWorkerCall.cancel
    original_result = _PreparedWorkerCall.result

    def tracked_cancel(attempt, error):
        cancelled_attempts.append(attempt)
        return original_cancel(attempt, error)

    def tracked_result(attempt):
        awaited_attempts.append(attempt)
        return original_result(attempt)

    monkeypatch.setattr(_PreparedWorkerCall, "cancel", tracked_cancel)
    monkeypatch.setattr(_PreparedWorkerCall, "result", tracked_result)

    def close_rows(phase, envelope):
        if phase == "MPI observer queue finalization enqueue":
            assert queue._finalize_attempt is not None
            prepared_attempts.append(queue._finalize_attempt)
            raise RuntimeError("injected finalization enqueue consensus failure")
        peer = dict(envelope)
        peer["rank"] = 1
        return envelope, peer

    publisher._collective_close_rows = close_rows

    def peer_flush(_communicator, envelope):
        return envelope, {"rank": 1, "reports": [], "diagnostics": []}

    monkeypatch.setattr(_runtime_consumers, "allgather_value", peer_flush)

    try:
        with pytest.raises(
            _runtime_consumers._ObserverCollectiveLost,
            match="finalization enqueue consensus failed",
        ):
            publisher._drain_observer_manifest(manifest, run_identity, close=True)
        assert len(prepared_attempts) == 1
        assert cancelled_attempts == prepared_attempts
        assert awaited_attempts == prepared_attempts
        assert prepared_attempts[0]._done.is_set()
        assert queue._finalize_attempt is None
        assert session.finalize_calls == 0
        assert key in publisher._observer_queues
        assert key in publisher._observer_lanes
        assert lane.close_calls == 0
        assert lane.closed is False
        assert publisher._observer_finalize_retry_blocked == set()
    finally:
        cleanup()


def test_mpi_world_report_consensus_loss_is_sticky_and_keeps_reports_private(monkeypatch):
    from pops.runtime import _runtime_consumers
    from pops.runtime._observer_runtime import ObserverDeliveryReport
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    run_identity = make_identity("run", {"case": "retained-finalized-reports"})
    manifest = SimpleNamespace(
        kind=ConsumerKind.MONITOR,
        qualified_id="monitor/retained-finalized-reports",
        parallel_mode=ParallelMode.COLLECTIVE,
        operation_data={"on_failure": {"action": "raise_on_flush"}},
    )
    key = (manifest.qualified_id, run_identity.token)
    frame_identity = make_identity(
        "post-commit-observer-frame",
        {"case": "retained-finalized-reports"},
    )
    report = ObserverDeliveryReport(
        manifest.qualified_id,
        run_identity,
        0,
        frame_identity,
        "delivered",
        1,
        receipt=ObserverReceipt(frame_identity, "test.collective-finalize"),
    )

    class _Queue:
        close_requested = False
        close_succeeded = False
        abort_required = False
        reports = (report,)

        def __init__(self):
            self.finalize_calls = 0

        def prepare_close(self):
            self.close_requested = True
            return self.reports

        def prepare_complete_close(self):
            return None

        def cancel_complete_close(self, _error):
            return None

        def arm_complete_close(self):
            return None

        def complete_close(self):
            self.finalize_calls += 1
            self.close_succeeded = True
            return self.reports

    class _Lane:
        def __init__(self):
            self.closed = False
            self.close_calls = 0

        def close_collectively(self):
            self.close_calls += 1
            self.closed = True

    queue = _Queue()
    lane = _Lane()
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 2
    publisher._communicator = SimpleNamespace(identity="MPI_COMM_WORLD")
    publisher._observer_run_phases = {run_identity.token: "closing_open"}
    publisher._observer_pending_sessions = {}
    publisher._observer_queues = {key: queue}
    publisher._observer_lanes = {key: lane}
    publisher._observer_pending_reports = {}
    publisher._observer_pending_failures = {}
    publisher._observer_abort_retry_blocked = set()
    publisher._observer_finalize_retry_blocked = set()
    publisher._observer_reports = {}
    publisher._observer_diagnostics = []
    assert publisher.post_commit_reports == ()

    def close_rows(_phase, envelope):
        peer = dict(envelope)
        peer["rank"] = 1
        return envelope, peer

    publisher._collective_close_rows = close_rows
    report_consensus_calls = 0

    def fail_report_consensus(_communicator, envelope):
        nonlocal report_consensus_calls
        assert set(envelope) == {"rank", "reports", "diagnostics"}
        report_consensus_calls += 1
        raise RuntimeError("injected report consensus loss")

    monkeypatch.setattr(_runtime_consumers, "allgather_value", fail_report_consensus)

    with pytest.raises(
        _runtime_consumers._ObserverCollectiveLost,
        match="flush lost its collective proof",
    ):
        publisher._drain_observer_manifest(manifest, run_identity, close=True)

    assert queue.finalize_calls == 1
    assert key not in publisher._observer_queues
    assert publisher._observer_pending_reports[key] == (report,)
    assert report.identity.token not in publisher._observer_reports
    assert publisher.post_commit_reports == ()
    assert publisher._observer_lanes[key] is lane
    assert lane.closed is True
    assert lane.close_calls == 1
    assert report_consensus_calls == 1
    assert "flush lost its collective proof" in publisher._observer_world_collective_lost

    with pytest.raises(RuntimeError, match="MPI_COMM_WORLD is sealed"):
        publisher._drain_observer_manifest(manifest, run_identity, close=True)

    assert report_consensus_calls == 1
    assert queue.finalize_calls == 1
    assert publisher._observer_pending_reports[key] == (report,)
    assert publisher._observer_reports == {}
    assert publisher.post_commit_reports == ()
    assert publisher._observer_lanes[key] is lane
    assert lane.close_calls == 1


def test_observer_drain_accepts_recovery_run_identity_and_refuses_foreign_run():
    from pops.runtime._observer_runtime import ObserverDeliveryReport, ObserverRun
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    run_identity = make_identity("run", {"case": "active-report-authority"})
    recovery_identity = make_identity("run", {"case": "recovery-report-authority"})
    foreign_identity = make_identity("run", {"case": "foreign-report-authority"})
    manifest = SimpleNamespace(
        kind=ConsumerKind.MONITOR,
        qualified_id="monitor/recovery-report-authority",
        parallel_mode=ParallelMode.SERIAL,
        operation_data={"on_failure": {"action": "raise_on_flush"}},
    )
    key = (manifest.qualified_id, run_identity.token)

    def delivered_report(identity, sequence):
        frame_identity = make_identity(
            "post-commit-observer-frame",
            {"run": identity.token, "sequence": sequence},
        )
        return ObserverDeliveryReport(
            manifest.qualified_id,
            identity,
            sequence,
            frame_identity,
            "delivered",
            1,
            receipt=ObserverReceipt(frame_identity, "test.recovery-report-authority"),
        )

    recovery_report = delivered_report(recovery_identity, 0)
    foreign_report = delivered_report(foreign_identity, 1)

    class _Queue:
        close_requested = False
        close_succeeded = False

        def __init__(self):
            self.reports = (recovery_report,)

        def flush(self):
            return self.reports

    queue = _Queue()
    observer_run = ObserverRun(
        run_identity,
        recovery_run_identities=(recovery_identity,),
    )
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 1
    publisher._communicator = None
    publisher._observer_run_phases = {run_identity.token: "open"}
    publisher._observer_pending_sessions = {}
    publisher._observer_queues = {key: queue}
    publisher._observer_lanes = {}
    publisher._observer_pending_reports = {}
    publisher._observer_report_run_authorities = {
        key: frozenset(observer_run.accepted_run_identities)
    }
    publisher._observer_pending_failures = {}
    publisher._observer_reports = {}
    publisher._observer_diagnostics = []

    assert publisher._drain_observer_manifest(manifest, run_identity, close=False) == ()
    assert publisher._observer_reports[recovery_report.identity.token] == recovery_report

    queue.reports = (foreign_report,)
    with pytest.raises(RuntimeError, match="authenticates another run or session"):
        publisher._drain_observer_manifest(manifest, run_identity, close=False)
    assert foreign_report.identity.token not in publisher._observer_reports


def test_close_preflight_refuses_worker_missing_on_one_mpi_rank(monkeypatch):
    from pops.runtime import _runtime_consumers
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    run_identity = make_identity("run", {"case": "missing-rank-worker"})
    manifest = SimpleNamespace(
        kind=ConsumerKind.MONITOR,
        qualified_id="monitor/collective-worker",
        parallel_mode=ParallelMode.COLLECTIVE,
        identity=make_identity("consumer-manifest", {"case": "collective-worker"}),
        operation_data={"observer": {"provider": {"provider_id": "test.collective-observer"}}},
    )
    key = (manifest.qualified_id, run_identity.token)
    queue = SimpleNamespace(
        close_authority={
            "run_identity": run_identity.token,
            "consumer_id": manifest.qualified_id,
            "provider_id": "test.collective-observer",
        },
        close_requested=False,
        close_succeeded=False,
    )
    lane = SimpleNamespace(
        identity="MPI_COMM_WORLD/post-commit/%s/%s" % (manifest.identity.token, run_identity.token),
        active=True,
        closed=False,
    )
    worker = SimpleNamespace(
        close_authority=run_identity.token,
        close_requested=False,
        close_succeeded=False,
    )
    publisher = object.__new__(RuntimeConsumerPublisher)
    publisher._rank = 0
    publisher._size = 2
    publisher._communicator = SimpleNamespace(identity="MPI_COMM_WORLD")
    publisher._root_output_consumers = ()
    publisher._root_output_lanes = {}
    publisher._observer_run_phases = {run_identity.token: "open"}
    publisher._owner = SimpleNamespace(_consumer_graph=SimpleNamespace(nodes=(manifest,)))
    publisher._observer_workers = {run_identity.token: worker}
    publisher._observer_queues = {key: queue}
    publisher._observer_lanes = {key: lane}

    def missing_peer_worker(_communicator, envelope):
        peer = dict(envelope)
        peer["rank"] = 1
        peer["monitors"] = [dict(row) for row in envelope["monitors"]]
        peer["worker"] = None
        return envelope, peer

    monkeypatch.setattr(_runtime_consumers, "allgather_value", missing_peer_worker)
    with pytest.raises(RuntimeError, match="without every run worker"):
        publisher._preflight_observer_close(run_identity)


def test_diagnostic_component_requires_one_explicit_role_for_multicomponent_state():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    with pytest.raises(ValueError, match="explicit typed ComponentRole"):
        RuntimeConsumerPublisher._diagnostic_component(
            ("rho", "momentum_x"), ("Density", "MomentumX"), None
        )
    assert RuntimeConsumerPublisher._diagnostic_component(
        ("rho", "momentum_x"), ("Density", "MomentumX"), "Density"
    ) == (0, False)


def test_adaptive_diagnostic_passes_the_exact_selected_levels_to_native_provider():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    calls = []

    class _AdaptiveProvider:
        def composite_reduce(self, block, reduction, component, levels):
            calls.append((block, reduction, component, levels))
            return 4.5

    value, composite = RuntimeConsumerPublisher._native_diagnostic_reduction(
        SimpleNamespace(), _AdaptiveProvider(), "fluid", "sum", 1, False, (0, 2)
    )
    assert (value, composite) == (4.5, True)
    assert calls == [("fluid", "sum", 1, [0, 2])]


def test_step_change_diagnostic_uses_the_native_transaction_snapshot():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    class _Provider:
        def _step_change_l2(self):
            return {"fluid": 0.125}

    value, composite = RuntimeConsumerPublisher._native_diagnostic_reduction(
        SimpleNamespace(), _Provider(), "fluid", "step_change_l2", 0, True, (0, 1)
    )
    assert (value, composite) == (0.125, True)


def test_balance_diagnostic_accepts_only_the_exact_native_five_term_tuple():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    class _Provider:
        def _accepted_balance_terms(self, route):
            assert route == "pops.balance-ledger-route.v1:sha256:" + "1" * 64
            return {
                "storage_change": 11.0,
                "outward_boundary_flux": 2.0,
                "sources": 5.0,
                "reflux": 3.0,
                "projection": 1.0,
            }

    terms = RuntimeConsumerPublisher._native_balance_terms(
        _Provider(),
        "pops.balance-ledger-route.v1:sha256:" + "1" * 64,
        block="fluid",
        component=0,
        levels=(0,),
        automatic_terms=(),
    )
    assert terms.residual == 4.0
    assert terms.reflux == 3.0

    class _Incomplete:
        def _accepted_balance_terms(self, _route):
            return {"storage_change": 1.0}

    with pytest.raises(TypeError, match="exactly storage_change"):
        RuntimeConsumerPublisher._native_balance_terms(
            _Incomplete(),
            "route",
            block="fluid",
            component=0,
            levels=(0,),
            automatic_terms=(),
        )

    class _Coerced:
        def _accepted_balance_terms(self, _route):
            return {
                "storage_change": "1.0",
                "outward_boundary_flux": 2.0,
                "sources": 5.0,
                "reflux": 3.0,
                "projection": 1.0,
            }

    with pytest.raises(TypeError, match="exact floating-point"):
        RuntimeConsumerPublisher._native_balance_terms(
            _Coerced(),
            "route",
            block="fluid",
            component=0,
            levels=(0,),
            automatic_terms=(),
        )


def test_diagnostic_restart_restores_payload_terms_and_native_inspection_registry():
    from pops.identity import make_identity
    from pops.output.data import DiagnosticKey, DiagnosticPayload
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    payload = DiagnosticPayload(
        DiagnosticKey(
            Handle("mass", kind="diagnostic", owner=OwnerPath.consumer("restart-test")),
            make_identity("component-manifest", {"name": "fluid"}),
            make_identity("layout", {"name": "mesh"}),
            0,
            make_identity("consumer-diagnostic-quantity", {"name": "mass"}).token,
            "conservation:integral",
        ),
        0.125,
        "kg",
        {"quantity": 4.0, "baseline": 3.875},
    )
    source = object.__new__(RuntimeConsumerPublisher)
    source._baselines = {"baseline": 3.875}
    source._pending_baselines = {}
    source._diagnostics = {payload.key.identity.token: payload}
    source._pending = {}
    state = source.diagnostic_restart_state()

    recorded = {}
    executor = SimpleNamespace(record_program_diagnostic=recorded.__setitem__)
    restored = object.__new__(RuntimeConsumerPublisher)
    restored._owner = SimpleNamespace(_executor=executor)
    restored._baselines = {}
    restored._pending_baselines = {}
    restored._diagnostics = {}
    restored._pending = {}
    restored.restore_diagnostic_restart_state(state)

    assert restored.diagnostics == (payload,)
    assert restored._baselines == {"baseline": 3.875}
    assert recorded == {
        "%s:%s:%s"
        % (
            payload.key.reference.qualified_id,
            payload.key.reduction,
            payload.key.state_id,
        ): 0.125,
    }


def test_partial_diagnostic_publication_rolls_back_before_reporting_failure():
    from pops.identity import make_identity
    from pops.runtime._runtime_consumers import _PreparedDiagnostic

    effect = SimpleNamespace(
        identity=make_identity("accepted-side-effect-test", {"sample": 1}),
        payload=SimpleNamespace(identity=make_identity("consumer-payload-test", {"sample": 1})),
    )
    calls = []

    def publish(_effect, _values):
        calls.append("publish-partial")
        raise RuntimeError("recorder failed")

    prepared = _PreparedDiagnostic(
        effect,
        (),
        publish,
        lambda _effect: calls.append("discard"),
        lambda _effect, _values: calls.append("rollback"),
    )
    with pytest.raises(RuntimeError, match="recorder failed"):
        prepared.publish()
    assert calls == ["publish-partial", "rollback"]
    prepared.rollback()
    assert calls == ["publish-partial", "rollback"]


def test_checkpoint_consumer_serializes_its_post_accept_cursor(tmp_path):
    target = tmp_path / "accepted-checkpoint.npz"
    plan, _, manifest = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        target_uri=target,
        operation=RestartV3(),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    runtime._run(t_end=1.0, max_steps=1)

    with np.load(target, allow_pickle=False) as payload:
        cursors = payload["runtime_consumer_cursors"].item()
    assert '"committed_samples":1' in cursors
    restored = RuntimeInstance(plan, executor=_Executor(plan))
    restored.restart(target)
    assert restored.consumer_cursors.for_consumer(manifest.qualified_id).committed_samples == 1
    assert restored._executor._prepared_restart_bit_identical is False


def test_checkpoint_restart_propagates_bit_identical_policy_to_native_preflight(tmp_path):
    target = tmp_path / "bit-identical-checkpoint.npz"
    plan, _, _ = _with_graph(
        tmp_path,
        kind=ConsumerKind.CHECKPOINT,
        output_format=None,
        target_uri=target,
        operation=RestartV3(bit_identical=True),
    )
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._run(t_end=1.0, max_steps=1)

    restored = RuntimeInstance(plan, executor=_Executor(plan))
    restored.restart(target)

    assert restored._executor._prepared_restart_bit_identical is True


def test_checkpoint_refuses_a_different_consumer_graph_before_native_restore(tmp_path):
    plan, _, _ = _with_graph(tmp_path / "outputs")
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._run(t_end=1.0, max_steps=1)
    checkpoint = runtime.checkpoint(tmp_path / "restart")

    empty_graph = ConsumerGraph(())
    from pops.output._restart_provider import RestartAuthority

    empty_record = replace(
        plan.artifact.plan,
        consumer_graph=empty_graph,
        restart_authority=RestartAuthority.from_consumer_graph(empty_graph),
    )
    empty_artifact = CompiledSimulationArtifact(
        empty_record, plan.artifact.program, plan.artifact.blocks
    )
    inputs = BindInputs()
    empty_plan = InstallPlan(
        artifact=empty_artifact,
        bind_inputs=inputs,
        instances={
            block.name: {"model": block.model, "spatial": block.spatial}
            for block in empty_artifact.blocks
        },
        params=empty_artifact.bind_schema.resolve_bind(
            {}, compile_values=empty_artifact.plan.compile_values
        ),
        aux={},
        execution_context=artifact_execution_context(empty_artifact),
    )
    other = RuntimeInstance(empty_plan, executor=_Executor(empty_plan))
    try:
        other.restart(checkpoint)
    except ValueError as error:
        assert "ConsumerGraph identity" in str(error)
    else:
        raise AssertionError("different ConsumerGraph restart was accepted")
    assert other.time() == 0.0
