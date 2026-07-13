"""One installed runtime for Uniform, AMR, multiblock and multi-layout plans."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from pops.codegen._plans import require_install_plan
from pops.fields import LayoutBinding
from pops.identity import Identity, make_identity
from pops.time import TimePoint

from ._consumer_contracts import (
    ConsumerCursorSet,
    ConsumerGraph,
    ConsumerKind,
    ConsumerMoment,
    ScheduleCursor,
    SkipSampleReported,
)
from ._consumer_planning import plan_accepted_side_effects
from ._consumer_transaction import ConsumerTransaction
from ._runtime_component_manifests import component_manifests_for_install
from ._runtime_consumers import RuntimeConsumerPublisher, RuntimeOutputSnapshot
from ._runtime_executor import install_runtime_executor
from ._runtime_planning import build_runtime_plans
from ._runtime_plan_io import thaw_data


_STRUCTURAL = frozenset({
    "add_block", "add_equation", "add_background", "add_coupling",
    "add_elliptic_model", "add_dynamic_block", "add_compiled_block", "add_native_block",
    "set_poisson", "install_program", "set_program_cadence", "set_refinement",
    "set_phi_refinement", "set_block_params", "set_program_params", "_install_compiled",
})


def _identity_data(value: Any) -> Any:
    """Detach an optional identity without retaining a live report/runtime object."""
    if value is None:
        return None
    to_data = getattr(value, "to_data", None)
    if callable(to_data):
        return to_data()
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("runtime identity evidence must implement to_data() or be canonical scalar data")


class RuntimeInstance:
    """Authenticated InstallPlan plus its sole native executor and transactional consumers."""

    def __init__(
        self,
        install_plan: Any,
        *,
        executor: Any = None,
        component_manifests: Any = None,
        publisher: Any = None,
    ) -> None:
        plan = require_install_plan(install_plan)
        manifests = component_manifests_for_install(plan) \
            if component_manifests is None else component_manifests
        runtime_plan = build_runtime_plans(plan, manifests)
        graph = plan.artifact.plan.consumer_graph
        if graph is None:
            graph = ConsumerGraph(())
        if type(graph) is not ConsumerGraph:
            raise TypeError("RuntimeInstance requires an exact ConsumerGraph")
        native = install_runtime_executor(plan) if executor is None else executor
        if native is None:
            raise TypeError("RuntimeInstance executor cannot be None")
        self.install_plan = plan
        self.layout_plan = plan.artifact.layout_plan
        self.execution_context = plan.execution_context
        self.component_manifests = manifests
        self.runtime_plan = runtime_plan
        self.consumer_graph = graph
        self.native_executor = native
        self.consumer_cursors = ConsumerCursorSet()
        self.consumer_reports = ()
        self.output_root = None
        self._attempt = 0
        self._checkpoint_cursor_override = None
        self._snapshot_builder = RuntimeOutputSnapshot(self)
        self._publisher = RuntimeConsumerPublisher(self) if publisher is None else publisher

    @property
    def bound_snapshot(self) -> Any:
        return self.native_executor.bound_snapshot

    @property
    def bind_identity(self) -> Identity:
        return self.install_plan.bind_identity

    @property
    def last_run_identity(self) -> Any:
        return getattr(self.native_executor, "last_run_identity", None)

    @property
    def last_restart_identity(self) -> Any:
        return getattr(self.native_executor, "last_restart_identity", None)

    def layout_identity(self, layout_id: str) -> Identity:
        rows = [row for row in self.layout_plan.layouts
                if row.handle.qualified_id == layout_id]
        if len(rows) != 1:
            raise KeyError("unknown RuntimeInstance layout %s" % layout_id)
        return make_identity("layout", rows[0].to_data())

    def output_snapshot(self, manifest: Any, diagnostics: Any = ()) -> Any:
        return self._snapshot_builder.build(manifest, tuple(diagnostics))

    def inspect(self) -> Any:
        """Return one array-free report spanning the accepted runtime and its install contract."""
        from pops.runtime.inspection import build_runtime_inspection

        layouts = tuple(self.layout_plan.layouts)
        adaptive = any(row.adaptive for row in layouts)
        return build_runtime_inspection(
            self.native_executor,
            runtime="adaptive" if adaptive else "uniform",
            adaptive=adaptive,
            instance={
                "bind_identity": self.bind_identity.to_data(),
                "artifact_identity": self.install_plan.artifact.artifact_identity.to_data(),
                "plan_identity": self.install_plan.artifact.plan.plan_identity.to_data(),
                "layout_plan": self.layout_plan.inspect(),
                "execution_context": self.execution_context.to_data(),
                "consumer_graph": self.consumer_graph.to_data(),
                "consumer_cursors": self.consumer_cursors.to_data(),
                "consumer_reports": [report.to_data() for report in self.consumer_reports],
                "attempt": self._attempt,
                "output_root": None if self.output_root is None else str(self.output_root),
                "last_run_identity": _identity_data(self.last_run_identity),
                "last_restart_identity": _identity_data(self.last_restart_identity),
            },
        )

    def _layout_bindings(self) -> tuple[LayoutBinding, ...]:
        generation = 0
        counter = getattr(self.native_executor, "checkpoint_topology_epoch", None)
        if callable(counter):
            generation = int(counter())
        return tuple(LayoutBinding(row.handle, generation) for row in self.layout_plan.layouts)

    def _moments(self, *, at_start: bool = False, at_end: bool = False) -> tuple[ConsumerMoment, ...]:
        clocks = {row.schedule.domain.clock for row in self.consumer_graph.nodes}
        step = int(self.native_executor.macro_step())
        return tuple(
            ConsumerMoment(
                TimePoint(clock, step=step),
                accepted_step=step,
                attempt=self._attempt,
                clock_tick=step,
                wall_tick=step,
                layouts=self._layout_bindings(),
                at_start=at_start,
                at_end=at_end,
            )
            for clock in sorted(clocks, key=lambda value: value.qualified_id)
        )

    def _fire_consumers(self, *, at_start: bool = False, at_end: bool = False) -> tuple[Any, ...]:
        reports = []
        for moment in self._moments(at_start=at_start, at_end=at_end):
            effects = plan_accepted_side_effects(
                self.runtime_plan, self.consumer_graph, moment, self.consumer_cursors)
            checkpoint_ids = {
                row.qualified_id for row in self.consumer_graph.nodes
                if row.kind is ConsumerKind.CHECKPOINT
            }
            checkpoint_effects = tuple(
                row for row in effects.effects if row.consumer_id in checkpoint_ids)
            if checkpoint_effects:
                if len(checkpoint_effects) != 1 or effects.effects[-1] is not checkpoint_effects[0]:
                    raise ValueError(
                        "an accepted moment may publish exactly one checkpoint and it must be "
                        "the final ConsumerGraph effect"
                    )
                if any(type(row.failure_action) is SkipSampleReported for row in effects.effects):
                    raise ValueError(
                        "a checkpoint transaction cannot predict restart cursors when another "
                        "effect may skip its sample"
                    )
                predicted = self.consumer_cursors
                for row in effects.effects:
                    predicted = predicted.replace(row.cursor_after)
                self._checkpoint_cursor_override = predicted
            try:
                transaction = ConsumerTransaction(
                    effects, self.consumer_cursors, self._publisher)
                report = transaction.accept()
            finally:
                self._checkpoint_cursor_override = None
            self.consumer_cursors = report.cursors
            reports.append(report)
        self.consumer_reports = self.consumer_reports + tuple(reports)
        return tuple(reports)

    def _run(self, t_end: Any, max_steps: int = 1_000_000,
             output_dir: Any = None, **controller_controls: Any) -> int:
        from pops.runtime._step_strategy import (
            AdaptiveCFL, run_control_payload, run_step_attempt)

        native = self.native_executor
        program = self.install_plan.artifact.plan.time
        selected = getattr(program, "_step_strategy", None)
        if selected is None:
            raise ValueError(
                "pops.run requires the Program to declare step_strategy(...); runtime kwargs "
                "cannot select a numerical controller")
        program.validate_runtime_controls(controller_controls)
        control = run_control_payload(selected)
        temporal = getattr(native, "_temporal_restart_state", None)
        if temporal is not None:
            temporal.begin_run(control, time=native.time(), macro_step=native.macro_step())
        from pops.runtime._run_manifest import begin_run

        manifest_cfl = control["cfl"] if isinstance(selected, AdaptiveCFL) else 0.0
        begin_run(native, t_end=t_end, cfl=manifest_cfl,
                  max_steps=max_steps, output_dir=output_dir)
        previous_root, self.output_root = self.output_root, output_dir
        steps = 0
        try:
            self._fire_consumers(at_start=True)
            while native.time() < t_end and steps < max_steps:
                self._attempt += 1
                run_step_attempt(native, native, selected, t_end=float(t_end))
                steps += 1
                at_end = not (native.time() < t_end) or steps >= max_steps
                self._fire_consumers(at_end=at_end)
            if steps == 0:
                self._fire_consumers(at_end=True)
        finally:
            self.output_root = previous_root
        return steps

    def step(self, dt: Any) -> Any:
        self._attempt += 1
        result = self.native_executor.step(dt)
        self._fire_consumers()
        return result

    def step_cfl(self, cfl: Any) -> Any:
        self._attempt += 1
        result = self.native_executor.step_cfl(cfl)
        self._fire_consumers()
        return result

    def _checkpoint_payload(self, path: Any) -> str:
        target = self.native_executor.checkpoint(str(path))
        import numpy as np
        from ._checkpoint_manifest import IDENTITY_KEY, MANIFEST_KEY, seal_checkpoint_payload

        with np.load(target, allow_pickle=False) as stored:
            old_manifest = json.loads(str(stored[MANIFEST_KEY]))
            payload = {
                name: np.asarray(stored[name]).copy()
                for name in stored.files if name not in {MANIFEST_KEY, IDENTITY_KEY}
            }
        payload["runtime_consumer_graph"] = np.asarray(self.consumer_graph.identity.token)
        cursors = self._checkpoint_cursor_override or self.consumer_cursors
        payload["runtime_consumer_cursors"] = np.asarray(json.dumps(
            cursors.to_data(), sort_keys=True, separators=(",", ":")))
        seal_checkpoint_payload(self, payload, runtime_kind=old_manifest["runtime_kind"])
        temporary = str(target) + ".runtime-instance.tmp"
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, target)
        return target

    def _restart_operation(self) -> Any:
        manifests = tuple(
            row for row in self.consumer_graph.nodes if row.kind is ConsumerKind.CHECKPOINT)
        if not manifests:
            from .restart_provider import RestartV3

            return RestartV3()
        evidence = {
            make_identity("restart-provider", thaw_data(row.operation_data)).token
            for row in manifests
        }
        if len(evidence) != 1:
            raise ValueError("RuntimeInstance has multiple incompatible restart providers")
        return manifests[0].operation

    def checkpoint(self, path: Any) -> str:
        target = str(path)
        operation = self._restart_operation()
        extension = operation.consumer_data()["extension"]
        if not target.endswith(extension):
            target += extension
        snapshot = operation.snapshot(self, os.path.dirname(target) or ".")
        return str(operation.write(snapshot, target))

    def _reopen_checkpoint(self, path: Any) -> tuple[str, ConsumerCursorSet]:
        import numpy as np

        target = str(path) if str(path).endswith(".npz") else str(path) + ".npz"
        with np.load(target, allow_pickle=False) as stored:
            required = {"runtime_consumer_graph", "runtime_consumer_cursors"}
            if not required <= set(stored.files):
                raise ValueError("restart checkpoint lacks RuntimeInstance consumer state")
            if str(stored["runtime_consumer_graph"]) != self.consumer_graph.identity.token:
                raise ValueError("restart ConsumerGraph identity differs from the installed graph")
            cursor_data = json.loads(str(stored["runtime_consumer_cursors"]))
        if set(cursor_data) != {"schema_version", "rows"} or cursor_data["schema_version"] != 1:
            raise ValueError("restart consumer cursor schema is unsupported")
        cursors = ConsumerCursorSet(tuple(ScheduleCursor(
            row["consumer_id"], row["last_occurrence"], row["committed_samples"])
            for row in cursor_data["rows"]))
        return target, cursors

    def _restore_checkpoint(self, target: Any, cursors: ConsumerCursorSet) -> Any:
        result = self.native_executor.restart(str(target))
        self.consumer_cursors = cursors
        return result

    def restart(self, path: Any) -> Any:
        operation = self._restart_operation()
        reopened = operation.reopen(self, path)
        return operation.restore(self, reopened)

    def __getattr__(self, name: str) -> Any:
        if name in _STRUCTURAL:
            raise AttributeError(
                "%r is compile/install vocabulary and is unavailable on RuntimeInstance" % name)
        return getattr(self.native_executor, name)

    def __str__(self) -> str:
        return "RuntimeInstance(layouts=%d, blocks=%d, consumers=%d)" % (
            len(self.layout_plan.layouts), len(self.install_plan.instances),
            len(self.consumer_graph.nodes))


__all__ = ["RuntimeInstance"]
