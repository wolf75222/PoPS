"""One installed runtime for Uniform, AMR, multiblock and multi-layout plans."""
from __future__ import annotations

import copy
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
from ._runtime_consumers import RuntimeConsumerPublisher, RuntimeOutputSnapshot, _layout_identity
from ._runtime_executor import install_runtime_executor
from ._runtime_planning import build_runtime_plans
from ._runtime_plan_io import thaw_data


_STRUCTURAL = frozenset({
    "add_block", "add_equation", "add_background", "add_coupling",
    "add_elliptic_model", "add_dynamic_block", "add_compiled_block", "add_native_block",
    "set_poisson", "install_program", "set_refinement",
    "set_phi_refinement", "set_block_params", "set_program_params", "_install_compiled",
})
_RETIRED_EXECUTION_BYPASSES = frozenset({"run", "step", "step_cfl"})


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
        return _layout_identity(rows[0])

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
        native = self.native_executor
        temporal = getattr(native, "_temporal_restart_state", None)
        if clocks and temporal is None:
            raise RuntimeError(
                "RuntimeInstance consumers require accepted qualified temporal clock state")
        accepted_step = int(native.macro_step())
        moments = []
        for clock in sorted(clocks, key=lambda value: value.qualified_id):
            cursor = temporal.cursor_for_clock(clock)
            moments.append(ConsumerMoment(
                TimePoint(clock, step=int(cursor["tick"])),
                accepted_step=accepted_step,
                attempt=self._attempt,
                clock_tick=int(cursor["tick"]),
                wall_tick=accepted_step,
                layouts=self._layout_bindings(),
                at_start=at_start,
                at_end=at_end,
            ))
        return tuple(moments)

    def _stage_consumers(
        self, *, at_start: bool = False, at_end: bool = False,
    ) -> tuple[ConsumerTransaction, ...]:
        plans = tuple(
            plan_accepted_side_effects(
                self.runtime_plan, self.consumer_graph, moment, self.consumer_cursors)
            for moment in self._moments(at_start=at_start, at_end=at_end)
        )
        plans = tuple(plan for plan in plans if plan.effects)
        all_effects = tuple(effect for plan in plans for effect in plan.effects)
        checkpoint_ids = {
            row.qualified_id for row in self.consumer_graph.nodes
            if row.kind is ConsumerKind.CHECKPOINT
        }
        checkpoint_effects = tuple(
            effect for effect in all_effects if effect.consumer_id in checkpoint_ids)
        if checkpoint_effects:
            if len(checkpoint_effects) != 1 or all_effects[-1] is not checkpoint_effects[0]:
                raise ValueError(
                    "an accepted transaction may publish exactly one checkpoint and it must be "
                    "the final ConsumerGraph effect"
                )
            if any(type(effect.failure_action) is SkipSampleReported for effect in all_effects):
                raise ValueError(
                    "a checkpoint transaction cannot predict restart cursors when another "
                    "effect may skip its sample"
                )
            predicted = self.consumer_cursors
            for effect in all_effects:
                predicted = predicted.replace(effect.cursor_after)
            self._checkpoint_cursor_override = predicted

        staged = []
        try:
            for plan in plans:
                staged.append(ConsumerTransaction(
                    plan, self.consumer_cursors, self._publisher))
        except BaseException as error:
            cleanup_error = self._abort_consumers(tuple(staged))
            if cleanup_error is not None:
                raise cleanup_error from error
            raise
        finally:
            self._checkpoint_cursor_override = None
        return tuple(staged)

    @staticmethod
    def _abort_consumers(transactions: tuple[ConsumerTransaction, ...]) -> BaseException | None:
        failure = None
        for transaction in reversed(transactions):
            try:
                transaction.abort()
            except BaseException as error:
                if failure is None:
                    failure = error
        return failure

    def _accept_consumers(
        self, transactions: tuple[ConsumerTransaction, ...],
    ) -> tuple[tuple[Any, ...], ConsumerCursorSet, tuple[Any, ...]]:
        reports = tuple(transaction.accept() for transaction in transactions)
        cursors = self.consumer_cursors
        for transaction in transactions:
            for cursor in transaction.cursor_updates:
                cursors = cursors.replace(cursor)
        return reports, cursors, self.consumer_reports + reports

    def _fire_consumers(self, *, at_start: bool = False, at_end: bool = False) -> tuple[Any, ...]:
        transactions = self._stage_consumers(at_start=at_start, at_end=at_end)
        try:
            reports, cursors, all_reports = self._accept_consumers(transactions)
        except BaseException as error:
            cleanup_error = self._abort_consumers(transactions)
            if cleanup_error is not None:
                raise cleanup_error from error
            raise
        self.consumer_cursors = cursors
        self.consumer_reports = all_reports
        for transaction in transactions:
            transaction.seal()
        return reports

    def _step_transaction_methods(self) -> tuple[Any, Any, Any, Any]:
        native = self.native_executor
        methods = tuple(getattr(native, name, None) for name in (
            "_begin_step_transaction", "_commit_step_transaction",
            "_finalize_step_transaction", "_rollback_step_transaction",
        ))
        if any(not callable(method) for method in methods):
            raise TypeError(
                "RuntimeInstance executor must implement the native step-transaction protocol"
            )
        return methods

    def _step_envelope_snapshot(self) -> dict[str, Any]:
        native = self.native_executor
        return {
            "attempt": self._attempt,
            "consumer_cursors": self.consumer_cursors,
            "consumer_reports": self.consumer_reports,
            "checkpoint_cursor_override": self._checkpoint_cursor_override,
            "temporal_restart_state": copy.deepcopy(
                getattr(native, "_temporal_restart_state", None)),
            "step_controller": copy.deepcopy(getattr(native, "_step_controller", None)),
            "last_step_transaction_report": getattr(
                native, "_last_step_transaction_report", None),
        }

    def _restore_step_envelope(self, snapshot: dict[str, Any]) -> None:
        native = self.native_executor
        self._attempt = snapshot["attempt"]
        self.consumer_cursors = snapshot["consumer_cursors"]
        self.consumer_reports = snapshot["consumer_reports"]
        self._checkpoint_cursor_override = snapshot["checkpoint_cursor_override"]
        if hasattr(native, "_temporal_restart_state"):
            native._temporal_restart_state = snapshot["temporal_restart_state"]
        if hasattr(native, "_step_controller"):
            native._step_controller = snapshot["step_controller"]
        if hasattr(native, "_last_step_transaction_report"):
            native._last_step_transaction_report = snapshot["last_step_transaction_report"]

    def _accepted_step_transaction(
        self,
        advance: Any,
        *,
        at_end: Any = False,
    ) -> Any:
        """Advance once and publish its due effects as one rollback boundary."""
        from pops.time import StepTransactionReport

        native = self.native_executor
        begin, commit, finalize, rollback = self._step_transaction_methods()
        snapshot = self._step_envelope_snapshot()
        phase = "begin"
        attempts = 1
        failure_report = None
        transactions = ()
        native_active = False
        begin_entered = False
        try:
            # ``begin`` is part of the failure boundary: a native backend may
            # have captured or mutated provisional stores before reporting a
            # failure.  Once entered, rollback must therefore be attempted even
            # when ``begin`` itself does not return.
            begin_entered = True
            begin()
            native_active = True
            phase = "solve"
            result, attempts = advance()
            if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
                raise RuntimeError("step controller returned an invalid native-attempt count")
            phase = "effect"
            self._attempt += attempts
            transactions = self._stage_consumers(
                at_end=bool(at_end() if callable(at_end) else at_end))
            phase = "commit"
            commit()
            phase = "effect"
            reports, cursors, all_reports = self._accept_consumers(transactions)
            phase = "commit"
            finalize()
            native_active = False
            self.consumer_cursors = cursors
            self.consumer_reports = all_reports
            for transaction in transactions:
                transaction.seal()
            return result
        except BaseException as error:
            failure_report = getattr(native, "_last_step_transaction_report", None)
            cleanup_error = self._abort_consumers(transactions)
            try:
                if native_active or begin_entered:
                    try:
                        rollback()
                    except BaseException as rollback_error:
                        # Preserve the initiating error.  Backends are required
                        # to make rollback safe after entering begin; surfacing
                        # that secondary contract violation as a note avoids
                        # replacing the actionable root cause.
                        if hasattr(error, "add_note"):
                            error.add_note(
                                "step-transaction rollback also failed: "
                                f"{rollback_error}")
            finally:
                self._restore_step_envelope(snapshot)
            if phase in {"effect", "commit"}:
                stores = tuple(
                    store.value for store in getattr(
                        native, "_step_transaction_plan", ()).stores
                ) if getattr(native, "_step_transaction_plan", None) is not None else ()
                failure_report = StepTransactionReport(
                    status="failed",
                    phase=phase,
                    action="fail_run",
                    attempts=attempts,
                    staged_effects=stores,
                    rolled_back_effects=stores,
                    diagnostics=(str(error),),
                )
            if failure_report is not None and hasattr(native, "_last_step_transaction_report"):
                native._last_step_transaction_report = failure_report
            if cleanup_error is not None:
                raise cleanup_error from error
            raise

    def _run(self, t_end: Any, *, max_steps: int = 1_000_000,
             output_dir: Any = None, **controller_controls: Any) -> int:
        from pops.runtime._step_strategy import (
            prepare_step_controller, resolve_run_strategy, run_control_payload, run_step_attempt)

        native = self.native_executor
        selected = resolve_run_strategy(native)
        control = run_control_payload(selected, controller_controls)
        prepare_step_controller(native, selected, controller_controls)
        self._step_transaction_methods()
        temporal = getattr(native, "_temporal_restart_state", None)
        if temporal is not None:
            temporal.begin_run(control, time=native.time(), macro_step=native.macro_step())
        from pops.runtime._run_manifest import begin_run

        begin_run(native, t_end=t_end, step_transaction=control,
                  max_steps=max_steps, output_dir=output_dir)
        previous_root, self.output_root = self.output_root, output_dir
        steps = 0
        try:
            self._fire_consumers(at_start=True)
            while native.time() < t_end and steps < max_steps:
                def advance() -> tuple[Any, int]:
                    report = run_step_attempt(
                        native, native, selected, t_end=float(t_end),
                        controls=controller_controls)
                    return report, report.attempts

                self._accepted_step_transaction(
                    advance,
                    at_end=lambda: not (native.time() < t_end),
                )
                steps += 1
            if native.time() < t_end:
                raise RuntimeError(
                    "max_steps exhausted before t_end: "
                    f"accepted {steps} step(s), reached t={native.time()!r}, "
                    f"requested t_end={t_end!r}")
            if steps == 0:
                self._fire_consumers(at_end=True)
        finally:
            self.output_root = previous_root
        return steps

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
        if name in _RETIRED_EXECUTION_BYPASSES:
            raise AttributeError(
                "%r is not a RuntimeInstance operation; execute the installed "
                "Program.step_strategy(...) contract through pops.run(...)" % name)
        if name in _STRUCTURAL:
            raise AttributeError(
                "%r is compile/install vocabulary and is unavailable on RuntimeInstance" % name)
        return getattr(self.native_executor, name)

    def __str__(self) -> str:
        return "RuntimeInstance(layouts=%d, blocks=%d, consumers=%d)" % (
            len(self.layout_plan.layouts), len(self.install_plan.instances),
            len(self.consumer_graph.nodes))


__all__ = ["RuntimeInstance"]
