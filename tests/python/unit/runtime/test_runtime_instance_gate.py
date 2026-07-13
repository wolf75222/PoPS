"""ADC-687: one installed runtime and accepted-only exact consumers."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pops
import pytest

from pops.codegen._plans import BindInputs, InstallPlan
from pops.codegen.compiled_artifact import CompiledSimulationArtifact
from pops.model import Handle, OwnerPath
from pops.output import NPZ, NPZWriter, read_npz
from pops.runtime.consumer import (
    ConsumerGraph,
    ConsumerKind,
    ConsumerManifest,
    ConsumerQuantity,
    ParallelMode,
)
from pops.runtime.runtime_instance import RuntimeInstance
from pops.runtime.restart_provider import RestartV3
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import AcceptedStep, AtEnd, Clock, Every, FixedDt, Schedule
from tests.python.unit.runtime.test_runtime_planning import _install


class _Executor:
    def __init__(self, plan: InstallPlan) -> None:
        self._plan = plan
        self._s = self
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
        return 2

    def ny(self):
        return 2

    def block_names(self):
        return ["fluid"]

    def variable_names(self, block, space):
        assert block == "fluid" and space == "conservative"
        return ["rho"]

    def state_global(self, block):
        assert block == "fluid"
        return np.full(4, self._step + 1.0)

    def local_boxes(self, block):
        assert block == "fluid"
        return [(0, 0, 1, 1)]

    def local_state(self, block, index):
        assert block == "fluid" and index == 0
        return self.state_global(block).reshape(1, 2, 2)

    def reduce_component(self, block, kind, component):
        assert (block, kind, component) == ("fluid", "sum", 0)
        return float(np.sum(self.state_global(block)))

    def checkpoint(self, path):
        from pops.runtime._checkpoint_manifest import seal_checkpoint_payload
        from pops.runtime.bricks import abi_key

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
        from pops.runtime._checkpoint_manifest import authenticate_checkpoint_payload

        with np.load(path, allow_pickle=False) as payload:
            self._last_restart_identity = authenticate_checkpoint_payload(
                self, payload, runtime_kind="uniform")
            self._time = float(payload["t"])
            self._step = int(payload["macro_step"])
        return self._last_restart_identity


class _CustomNPZ:
    __pops_ir_immutable__ = True

    def consumer_data(self):
        return {
            "schema_version": 1,
            "provider_id": "pops.test.custom-npz.v1",
            "extension": ".npz",
            "parallel_mode": "serial",
        }

    def writer(self):
        return NPZWriter()


def _with_graph(tmp_path, *, kind=ConsumerKind.SCIENTIFIC_OUTPUT,
                output_format=None, target_uri=None, operation=None, schedule=None):
    base = _install()
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
        NPZ() if output_format is None and kind is ConsumerKind.SCIENTIFIC_OUTPUT
        else output_format,
        ParallelMode.SERIAL,
        operation=operation,
    )
    graph = ConsumerGraph((manifest,))
    record = replace(base.artifact.plan, consumer_graph=graph)
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
    )
    return plan, graph, manifest


def test_runtime_instance_retains_complete_multilayout_plan_without_target_dispatch():
    plan = _install(("fluid", "solid"), heterogeneous=True)
    runtime = RuntimeInstance(plan, executor=object())

    assert runtime.layout_plan is plan.artifact.layout_plan
    assert runtime.runtime_plan.layout_plan_id == runtime.layout_plan.qualified_id
    assert len(runtime.runtime_plan.calls) == 2
    assert len(runtime.runtime_plan.communication.transfers) == 1
    assert runtime.runtime_plan.communication.transfers[0].provider_id == \
        runtime.layout_plan.mappings[0].provider_id


def test_runtime_instance_inspection_exposes_install_and_consumer_evidence():
    plan = _install()
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    report = runtime.inspect()
    payload = report.to_dict()
    assert payload["runtime"] == "uniform"
    assert payload["instance"]["bind_identity"] == plan.bind_identity.to_data()
    assert payload["instance"]["plan_identity"] == plan.artifact.plan.plan_identity.to_data()
    assert payload["instance"]["consumer_graph"] == runtime.consumer_graph.to_data()
    assert payload["instance"]["consumer_cursors"]["rows"] == []
    assert pops.inspect(runtime) == payload


def test_runtime_instance_has_one_authored_execution_route():
    plan = _install()
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    assert not hasattr(runtime, "step")
    assert not hasattr(runtime, "step_cfl")
    assert not hasattr(runtime, "run")
    with pytest.raises(TypeError, match="does not accept strategy= or cfl="):
        pops.run(runtime, t_end=1.0, max_steps=1, strategy=FixedDt(1.0))


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

    assert pops.run(runtime, t_end=1.0, max_steps=1) == 1

    cursor = runtime.consumer_cursors.for_consumer(manifest.qualified_id)
    assert cursor.committed_samples == 1
    outputs = tuple(tmp_path.glob("*.npz"))
    assert len(outputs) == 1
    reopened = read_npz(outputs[0])
    assert reopened.manifest["snapshot"]["clock"]["macro_step"] == 1
    assert reopened.manifest["snapshot"]["metadata"] == {
        "consumer_graph": graph.identity.token,
        "runtime_plan": runtime.runtime_plan.identity.token,
    }


def test_run_fails_explicitly_when_max_steps_cannot_reach_t_end(tmp_path):
    plan, _, manifest = _with_graph(
        tmp_path, schedule=lambda clock: Schedule(AtEnd(AcceptedStep(clock))))
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    with pytest.raises(RuntimeError, match="max_steps exhausted before t_end"):
        runtime.run(2.0, max_steps=1)

    assert runtime.time() == 1.0
    cursor = runtime.consumer_cursors.for_consumer(manifest.qualified_id)
    assert cursor.committed_samples == 0
    assert tuple(tmp_path.glob("*.npz")) == ()


def test_scientific_format_is_a_structural_provider_without_name_dispatch(tmp_path):
    plan, _, _ = _with_graph(tmp_path, output_format=_CustomNPZ())
    runtime = RuntimeInstance(plan, executor=_Executor(plan))

    pops.run(runtime, t_end=1.0, max_steps=1)

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


def test_checkpoint_restart_authenticates_and_restores_consumer_cursors(tmp_path):
    plan, _, manifest = _with_graph(tmp_path / "outputs")
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    pops.run(runtime, t_end=1.0, max_steps=1)
    checkpoint = runtime.checkpoint(tmp_path / "restart")

    restored = RuntimeInstance(plan, executor=_Executor(plan))
    restored.restart(checkpoint)

    assert restored.consumer_cursors.for_consumer(manifest.qualified_id) == \
        runtime.consumer_cursors.for_consumer(manifest.qualified_id)
    assert restored.time() == runtime.time()
    with np.load(checkpoint, allow_pickle=False) as payload:
        assert str(payload["runtime_consumer_graph"]) == runtime.consumer_graph.identity.token
        assert "runtime_consumer_cursors" in payload.files


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

    pops.run(runtime, t_end=1.0, max_steps=1)

    with np.load(target, allow_pickle=False) as payload:
        cursors = payload["runtime_consumer_cursors"].item()
    assert '"committed_samples":1' in cursors
    restored = RuntimeInstance(plan, executor=_Executor(plan))
    restored.restart(target)
    assert restored.consumer_cursors.for_consumer(manifest.qualified_id).committed_samples == 1


def test_checkpoint_refuses_a_different_consumer_graph_before_native_restore(tmp_path):
    plan, _, _ = _with_graph(tmp_path / "outputs")
    runtime = RuntimeInstance(plan, executor=_Executor(plan))
    pops.run(runtime, t_end=1.0, max_steps=1)
    checkpoint = runtime.checkpoint(tmp_path / "restart")

    empty_record = replace(plan.artifact.plan, consumer_graph=ConsumerGraph(()))
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
    )
    other = RuntimeInstance(empty_plan, executor=_Executor(empty_plan))
    try:
        other.restart(checkpoint)
    except ValueError as error:
        assert "ConsumerGraph identity" in str(error)
    else:
        raise AssertionError("different ConsumerGraph restart was accepted")
    assert other.time() == 0.0
