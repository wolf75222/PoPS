"""ADC-687: one installed runtime and accepted-only exact consumers."""
from __future__ import annotations

from dataclasses import replace
import json
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
    ConsumerGraph,
    ConsumerKind,
    ConsumerManifest,
    ConsumerQuantity,
    ParallelMode,
)
from pops.output._restart_provider import RestartV3
from pops.output.observers import ObserverReceipt
from pops.runtime._runtime_instance import RuntimeInstance
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import (
    AcceptedStep,
    AdaptiveCFL,
    AtEnd,
    Clock,
    Every,
    ExternalTimeGrid,
    FixedDt,
    Schedule,
    every_dt,
)
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.unit.runtime.test_runtime_planning import _artifact as _planning_artifact


def _install(
    names=("fluid",), *, heterogeneous=False, memory_spaces=("host",)
):
    """Build the planning fixture against the exact loaded native ABI and resources."""
    from pops import _pops

    template = _planning_artifact(
        names, heterogeneous=heterogeneous, memory_spaces=memory_spaces
    )
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
        params=artifact.bind_schema.resolve_bind(
            {}, compile_values=artifact.plan.compile_values
        ),
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
        self._last_run_identity = None
        self._last_restart_identity = None
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
        graph = plan.artifact.plan.consumer_graph
        clocks = sorted(
            {node.schedule.domain.clock for node in graph.nodes},
            key=lambda clock: clock.qualified_id,
        ) if graph is not None else []
        self._temporal_restart_state = TemporalRestartState()
        if clocks:
            if len(clocks) != 1:
                raise ValueError("test executor requires one authored consumer clock")
            clock = clocks[0]
            self._temporal_restart_state.configure_program({
                "schema_version": 1,
                "kind": "pops.temporal-program-schedule",
                "primary_clock": clock.qualified_id,
                "clocks": [{
                    "id": clock.qualified_id,
                    "descriptor": clock.to_data(),
                    "ticks_per_macro": 1,
                }],
                "subcycles": [], "synchronizations": [], "schedules": [], "histories": [],
            }, time=0.0, macro_step=0)

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
        return [(0, 0, self._nx - 1, self._ny - 1)]

    def _output_geometry_snapshot(self, origin, spacing, shape, cell_measure):
        assert tuple(shape) == (self._ny, self._nx)
        assert cell_measure == "pops://cell-measures/cartesian-area@1"
        valid = np.ones(shape, dtype=np.bool_)
        coverage = np.zeros(shape, dtype=np.bool_)
        volumes = np.full(shape, spacing[0] * spacing[1], dtype=np.float64)
        for value in (valid, coverage, volumes):
            value.setflags(write=False)
        return {
            "topology_epoch": 0,
            "boxes": ((0, 0, shape[0], shape[1]),),
            "valid_cells": valid,
            "coverage": coverage,
            "cell_volumes": volumes,
        }

    def local_state(self, block, index):
        assert block == "fluid" and index == 0
        return self.state_global(block).reshape(1, self._ny, self._nx)

    def output_state_local_pieces(self, block, level):
        assert block == "fluid" and level == 0
        return ({
            "lower": (0, 0),
            "upper": (self._ny, self._nx),
            "values": np.ascontiguousarray(
                self.state_global(block).reshape(1, self._ny, self._nx),
                dtype=np.float64,
            ),
            "global_box_index": 0,
            "owner_rank": 0,
            "replicated": False,
        },)

    def output_state_root_pieces(self, communicator, block, level):
        """Expose the exact singleton-world gather required by ROOT publication tests."""
        from pops._native_collectives import require_world, size

        expected = self._plan.execution_context.communicator
        if communicator is not expected.handle:
            raise ValueError("ROOT gather did not receive the installed communicator handle")
        native = require_world(communicator)
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
        }
        seal_checkpoint_payload(self, payload, runtime_kind="uniform")
        target = path if str(path).endswith(".npz") else str(path) + ".npz"
        with open(target, "wb") as stream:
            np.savez_compressed(stream, **payload)
        return target

    def restart(self, path):
        prepared = self._prepare_checkpoint_restart(Path(path).read_bytes())
        self._begin_checkpoint_restart()
        result = self._apply_checkpoint_restart(prepared)
        self._commit_checkpoint_restart()
        self._finalize_checkpoint_restart()
        return result

    def _prepare_checkpoint_restart(self, payload):
        from pops.output._checkpoint_collective import decode_checkpoint_bytes
        from pops.runtime._checkpoint_manifest import authenticate_checkpoint_payload

        stored = decode_checkpoint_bytes(payload)
        identity = authenticate_checkpoint_payload(self, stored, runtime_kind="uniform")
        return identity, float(stored["t"]), int(stored["macro_step"])

    def _begin_checkpoint_restart(self):
        self._begin_step_transaction()
        self._restart_identity_snapshot = self._last_restart_identity

    def _apply_checkpoint_restart(self, prepared):
        identity, self._time, self._step = prepared
        self._last_restart_identity = identity
        return identity

    def _commit_checkpoint_restart(self):
        self._commit_step_transaction()

    def _finalize_checkpoint_restart(self):
        self._finalize_step_transaction()
        del self._restart_identity_snapshot

    def _rollback_checkpoint_restart(self):
        self._rollback_step_transaction()
        self._last_restart_identity = self._restart_identity_snapshot
        del self._restart_identity_snapshot


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


def _with_graph(tmp_path, *, kind=ConsumerKind.SCIENTIFIC_OUTPUT,
                output_format=None, target_uri=None, operation=None, schedule=None):
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
    manifest = ConsumerManifest(
        Handle("density", kind="consumer", owner=OwnerPath.consumer("adc-687")),
        kind,
        (quantity,),
        Schedule(Every(AcceptedStep(clock), 1)) if schedule is None else schedule(clock),
        str(tmp_path) if target_uri is None else str(target_uri),
        NPZ(mode=parallel_mode)
        if output_format is None and kind is ConsumerKind.SCIENTIFIC_OUTPUT
        else output_format,
        parallel_mode if kind is ConsumerKind.SCIENTIFIC_OUTPUT else ParallelMode.SERIAL,
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
        instances={block.name: {"model": block.model, "spatial": block.spatial}
                   for block in artifact.blocks},
        params=artifact.bind_schema.resolve_bind(
            {}, compile_values=artifact.plan.compile_values),
        aux={},
        execution_context=artifact_execution_context(artifact),
    )
    return plan, graph, manifest


def test_runtime_instance_retains_complete_multilayout_plan_without_target_dispatch():
    plan = _install(("fluid", "solid"), heterogeneous=True)
    runtime = RuntimeInstance(plan, executor=object())

    assert runtime._layout_plan is plan.artifact.layout_plan
    assert runtime._runtime_plan.layout_plan_id == runtime._layout_plan.qualified_id
    assert len(runtime._runtime_plan.calls) == 2
    assert len(runtime._runtime_plan.communication.transfers) == 1
    assert runtime._runtime_plan.communication.transfers[0].provider_id == \
        runtime._layout_plan.mappings[0].provider_id


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
            for statement in tree.body if isinstance(statement, ast.ClassDef)
            for node in statement.body if isinstance(node, ast.FunctionDef)
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
    assert payload["instance"]["consumer_graph"] == runtime.consumer_graph.to_data()
    assert payload["instance"]["restart_authority"] == \
        plan.artifact.plan.restart_authority.to_data()
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


def test_runtime_instance_has_one_authored_execution_route():
    plan = _install()
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    declared_public = {
        name for name in RuntimeInstance.__dict__ if not name.startswith("_")
    }
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
    temporal.configure_program({
        "schema_version": 1,
        "kind": "pops.temporal-program-schedule",
        "primary_clock": macro.qualified_id,
        "clocks": [
            {"id": macro.qualified_id, "descriptor": macro.to_data(), "ticks_per_macro": 1},
            {"id": child.qualified_id, "descriptor": child.to_data(), "ticks_per_macro": 4},
        ],
        "subcycles": [{
            "node_id": 3, "parent_clock": macro.qualified_id,
            "child_clock": child.qualified_id, "count": 4,
        }],
        "synchronizations": [], "schedules": [], "histories": [],
    }, time=0.0, macro_step=0)
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
    temporal.configure_program({
        "schema_version": 1,
        "kind": "pops.temporal-program-schedule",
        "primary_clock": unrelated.qualified_id,
        "clocks": [{
            "id": unrelated.qualified_id,
            "descriptor": unrelated.to_data(),
            "ticks_per_macro": 1,
        }],
        "subcycles": [], "synchronizations": [], "schedules": [], "histories": [],
    }, time=0.0, macro_step=0)
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
            make_identity("scientific-output", {
                "selection": self._request.publication_identity.token,
            }),
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

    def __init__(self):
        self.writer_started = threading.Event()
        self.release_writer = threading.Event()
        self.paths = []

    def consumer_data(self):
        return {
            "schema_version": 1,
            "provider_id": "pops.test.blocking-async.v1",
            "format_name": "blocking-test",
            "extension": ".async",
            "parallel_mode": "serial",
        }

    def writer(self):
        return _BlockingWriter(self)


def test_async_scientific_output_overlaps_next_step_and_flushes_real_receipts(tmp_path):
    output_root = tmp_path / "async-output"
    output_root.mkdir()
    format_provider = _BlockingFormat()
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
    tmp_path, policy, raises,
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
    checkpoint = runtime.checkpoint(tmp_path / "physical-cadence-restart")

    restored = RuntimeInstance(plan, executor=_Executor(plan))
    restored.restart(checkpoint)
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
        tmp_path, schedule=lambda clock: Schedule(AtEnd(AcceptedStep(clock))))
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    with pytest.raises(RuntimeError, match="max_steps exhausted before t_end"):
        runtime._run(2.0, max_steps=1)

    assert runtime.time() == 1.0
    cursor = runtime.consumer_cursors.for_consumer(manifest.qualified_id)
    assert cursor.committed_samples == 0
    assert tuple(tmp_path.glob("*.npz")) == ()


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


def test_checkpoint_restart_authenticates_and_restores_consumer_cursors(tmp_path):
    plan, _, manifest = _with_graph(tmp_path / "outputs")
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    runtime._run(t_end=1.0, max_steps=1)
    checkpoint = runtime.checkpoint(tmp_path / "restart")

    restored = RuntimeInstance(plan, executor=_Executor(plan))
    restored.restart(checkpoint)

    assert restored.consumer_cursors.for_consumer(manifest.qualified_id) == \
        runtime.consumer_cursors.for_consumer(manifest.qualified_id)
    assert restored.time() == runtime.time()
    with np.load(checkpoint, allow_pickle=False) as payload:
        assert str(payload["runtime_consumer_graph"]) == runtime.consumer_graph.identity.token
        assert "runtime_consumer_cursors" in payload.files
        diagnostic_state = json.loads(str(payload["runtime_consumer_diagnostics"]))
    assert diagnostic_state == {
        "schema_version": 2, "baselines": {}, "diagnostics": [],
    }


def test_checkpoint_diagnostic_baseline_schema_is_finite_and_canonical():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    canonical = {
        "schema_version": 2,
        "baselines": {"diagnostic:integral": 1.25.hex()},
        "diagnostics": [],
    }
    assert RuntimeConsumerPublisher.validate_diagnostic_restart_state(canonical) == canonical
    with pytest.raises(ValueError, match="finite"):
        RuntimeConsumerPublisher.validate_diagnostic_restart_state({
            "schema_version": 2,
            "baselines": {"diagnostic:integral": "nan"},
            "diagnostics": [],
        })
    with pytest.raises(ValueError, match="canonical"):
        RuntimeConsumerPublisher.validate_diagnostic_restart_state({
            "schema_version": 2,
            "baselines": {"diagnostic:integral": "0x1.4p+0"},
            "diagnostics": [],
        })


def test_diagnostic_component_requires_one_explicit_role_for_multicomponent_state():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    with pytest.raises(ValueError, match="explicit typed ComponentRole"):
        RuntimeConsumerPublisher._diagnostic_component(
            ("rho", "momentum_x"), ("Density", "MomentumX"), None)
    assert RuntimeConsumerPublisher._diagnostic_component(
        ("rho", "momentum_x"), ("Density", "MomentumX"), "Density") == (0, False)


def test_adaptive_diagnostic_passes_the_exact_selected_levels_to_native_provider():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    calls = []

    class _AdaptiveProvider:
        def composite_reduce(self, block, reduction, component, levels):
            calls.append((block, reduction, component, levels))
            return 4.5

    value, composite = RuntimeConsumerPublisher._native_diagnostic_reduction(
        SimpleNamespace(), _AdaptiveProvider(), "fluid", "sum", 1, False, (0, 2))
    assert (value, composite) == (4.5, True)
    assert calls == [("fluid", "sum", 1, [0, 2])]


def test_step_change_diagnostic_uses_the_native_transaction_snapshot():
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    class _Provider:
        def _step_change_l2(self):
            return {"fluid": 0.125}

    value, composite = RuntimeConsumerPublisher._native_diagnostic_reduction(
        SimpleNamespace(), _Provider(), "fluid", "step_change_l2", 0, True, (0, 1))
    assert (value, composite) == (0.125, True)


def test_diagnostic_restart_restores_payload_terms_and_native_inspection_registry():
    from pops.identity import make_identity
    from pops.output.data import DiagnosticKey, DiagnosticPayload
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher

    payload = DiagnosticPayload(
        DiagnosticKey(
            Handle("mass", kind="diagnostic", owner=OwnerPath.consumer("restart-test")),
            make_identity("component-manifest", {"name": "fluid"}),
            make_identity("layout", {"name": "mesh"}),
            0, make_identity("consumer-diagnostic-quantity", {"name": "mass"}).token,
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
        "%s:%s:%s" % (
            payload.key.reference.qualified_id, payload.key.reduction, payload.key.state_id,
        ): 0.125,
    }


def test_partial_diagnostic_publication_rolls_back_before_reporting_failure():
    from pops.identity import make_identity
    from pops.runtime._runtime_consumers import _PreparedDiagnostic

    effect = SimpleNamespace(
        identity=make_identity("accepted-side-effect-test", {"sample": 1}),
        payload=SimpleNamespace(
            identity=make_identity("consumer-payload-test", {"sample": 1})),
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
        empty_record, plan.artifact.program, plan.artifact.blocks)
    inputs = BindInputs()
    empty_plan = InstallPlan(
        artifact=empty_artifact,
        bind_inputs=inputs,
        instances={block.name: {"model": block.model, "spatial": block.spatial}
                   for block in empty_artifact.blocks},
        params=empty_artifact.bind_schema.resolve_bind(
            {}, compile_values=empty_artifact.plan.compile_values),
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
