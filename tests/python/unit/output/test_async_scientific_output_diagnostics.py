"""Diagnostics carried by AsyncScientificOutput are captured before post-commit dispatch."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading

import pytest

from pops.codegen._compiled_artifact import CompiledSimulationArtifact
from pops.codegen._plans import BindInputs, InstallPlan
from pops.diagnostics import Balance, BalanceLedger, Integral
from pops.identity import Identity, make_identity
from pops.layouts import Uniform
from pops.mesh import normalize_layout_plan
from pops.model import Handle, OwnerKind, OwnerPath
from pops.output import (
    AsyncScientificOutput,
    ConsumerGraph,
    NPZ,
    OutputPublicationReceipt,
    ParallelMode,
)
from pops.output._consumer_authoring import ConsumerAuthoringNode
from pops.output._consumer_contracts import (
    ConsumerKind,
    ConsumerManifest,
    DiagnosticQuantity,
)
from pops.output._restart_provider import RestartAuthority
from pops.output._writers.common import writer_session_authority
from pops.problem.handles import BlockHandle
from pops.runtime._runtime_consumers import RuntimeConsumerPublisher
from pops.runtime._runtime_instance import RuntimeInstance
from pops.time import Clock, every
from tests.python.support.layout_plan import cartesian_grid
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.unit.runtime.test_consumer_authoring import _case
from tests.python.unit.runtime.test_runtime_instance_gate import (
    _Executor,
    _install,
    _scientific_output_mode,
)


def _resolved_async_balance():
    case, block, state = _case()
    clock = Clock("async-balance", owner=case.owner_path)
    schedule = every(2, clock=clock)
    ledger = BalanceLedger("async-mass")
    descriptor = AsyncScientificOutput(
        format=NPZ(),
        schedule=schedule,
        diagnostics=(Balance(ledger, block=block, cadence=schedule),),
        target="async/balance",
    )
    graph = ConsumerGraph.from_consumers((descriptor,))
    case.consumers(graph)
    import pops

    pops.validate(case)
    subjects = case.layout_subjects()
    layout = normalize_layout_plan(
        Uniform(cartesian_grid(n=8)),
        owner=case.owner_path.canonical(),
        states=subjects.states,
        fields=subjects.fields,
        blocks=subjects.blocks,
        handle_resolver=case.resolve,
    )
    return (
        descriptor,
        graph.resolve(case.resolve, layout, owner=case.owner_path.canonical()),
        block,
        case.resolve(block),
        case.resolve(state),
        schedule,
        ledger,
    )


def test_async_scientific_output_accepts_diagnostic_only_and_resolves_balance():
    descriptor, graph, declared_block, block, state, schedule, ledger = (
        _resolved_async_balance()
    )
    (manifest,) = graph.nodes

    assert descriptor.fields == ()
    assert descriptor.declaration_references() == (declared_block,)
    assert manifest.kind is ConsumerKind.MONITOR
    assert manifest.quantities == ()
    assert manifest.operation_data["observer"]["observer_kind"] == "async_scientific_output"
    assert manifest.schedule == schedule
    (quantity,) = manifest.diagnostic_quantities
    assert quantity.reference == state
    assert quantity.levels == (0,)
    assert quantity.execution["operations"] == (
        {
            "name": "balance",
            "reduction": "accepted_balance",
            "transform": "identity",
            "metric_weighted": False,
            "balance_route": ledger.route_identity(block).token,
        },
    )


def test_async_scientific_output_requires_a_field_or_diagnostic_and_matching_cadence():
    case, block, state = _case()
    clock = Clock("async-validation", owner=case.owner_path)
    schedule = every(2, clock=clock)

    with pytest.raises(ValueError, match="at least one field or diagnostic"):
        AsyncScientificOutput(
            format=NPZ(),
            schedule=schedule,
            target="async/empty",
        )
    with pytest.raises(ValueError, match="must use the same schedule"):
        AsyncScientificOutput(
            format=NPZ(),
            schedule=schedule,
            diagnostics=(
                Integral(block=block, cadence=every(3, clock=clock)),
            ),
            target="async/cadence-mismatch",
        )

    descriptor = AsyncScientificOutput(
        format=NPZ(),
        schedule=schedule,
        fields=(state,),
        diagnostics=(Integral(block=block, cadence=schedule),),
        target="async/field-and-diagnostic",
    )
    assert descriptor.declaration_references() == (state, block)
    assert descriptor.options()["n_diagnostics"] == 1


class _NonScientificObserver:
    __pops_ir_immutable__ = True

    def consumer_data(self):
        return {
            "schema_version": 1,
            "provider_id": "pops.test.forged-async-scientific-observer.v1",
            "observer_kind": "async_scientific_output",
        }

    def open_session(self, _execution_context):
        raise AssertionError("authoring validation must not open an observer session")


def test_generic_monitor_cannot_smuggle_diagnostic_providers():
    from pops.output import AllLevels, LiveVisualization

    case, block, state = _case()
    clock = Clock("generic-monitor", owner=case.owner_path)
    schedule = every(1, clock=clock)
    live = LiveVisualization(
        observer=_NonScientificObserver(),
        schedule=schedule,
        fields=(state,),
    )
    operation = live.consumer_authoring()[0].operation

    with pytest.raises(ValueError, match="only AsyncScientificOutput"):
        ConsumerAuthoringNode(
            label="invalid-monitor-diagnostic",
            kind=ConsumerKind.MONITOR,
            references=(state,),
            schedule=schedule,
            target_uri="live",
            output_format=None,
            parallel_mode=ParallelMode.SERIAL,
            levels=AllLevels(),
            operation=operation,
            diagnostics=(Integral(block=block),),
        )

    _, graph, _, _, resolved_state, _, _ = _resolved_async_balance()
    (valid_async_manifest,) = graph.nodes
    forged_operation = LiveVisualization(
        observer=_NonScientificObserver(),
        schedule=valid_async_manifest.schedule,
        fields=(resolved_state,),
    ).consumer_authoring()[0].operation
    with pytest.raises(ValueError, match="only ConsoleMonitor, ScientificOutput"):
        replace(valid_async_manifest, operation=forged_operation)


class _CapturingWriterSession:
    def __init__(self, owner, request, target: Path) -> None:
        self.authority = writer_session_authority("capturing-async", request, target)
        self.identity = Identity.from_token(self.authority["session_identity"])
        self._owner = owner
        self._request = request
        self._target = target

    def stage(self):
        self._owner.writer_started.set()
        if not self._owner.release_writer.wait(timeout=10):
            raise TimeoutError("capturing async writer was not released")

    def abort_prepare(self):
        return None

    def publish(self):
        self._target.parent.mkdir(parents=True, exist_ok=True)
        self._target.write_bytes(b"captured detached diagnostics\n")
        return OutputPublicationReceipt(
            self._target,
            "capturing-async",
            make_identity(
                "scientific-output",
                {"selection": self._request.publication_identity.token},
            ),
            self._request.publication_identity,
        )

    def rollback(self):
        self._target.unlink(missing_ok=True)

    def finalize(self):
        return None


class _CapturingWriter:
    format = "capturing-async"

    def __init__(self, owner) -> None:
        self._owner = owner

    def preflight(self, _execution_context):
        return {"schema_version": 1, "provider_id": "capturing-async"}

    def prepare_session(self, snapshot, request, target, *, communicator=None):
        assert communicator is None
        self._owner.worker_threads.append(threading.current_thread().name)
        self._owner.snapshots.append(snapshot)
        return _CapturingWriterSession(self._owner, request, Path(target))


class _CapturingFormat:
    __pops_ir_immutable__ = True

    def __init__(self, mode: ParallelMode) -> None:
        self.mode = mode
        self.writer_started = threading.Event()
        self.release_writer = threading.Event()
        self.worker_threads: list[str] = []
        self.snapshots = []

    def consumer_data(self):
        return {
            "schema_version": 1,
            "provider_id": "pops.test.capturing-async.v1",
            "format_name": "capturing-async",
            "extension": ".capture",
            "parallel_mode": self.mode.value,
        }

    def writer(self):
        return _CapturingWriter(self)


class _BalanceExecutor(_Executor):
    def __init__(self, plan):
        super().__init__(plan)
        self.mailbox_open = True
        self.mailbox_calls: list[tuple[str, str]] = []

    def _accepted_balance_terms(self, route):
        if not self.mailbox_open:
            raise RuntimeError("post-commit worker attempted to read the native balance mailbox")
        self.mailbox_calls.append((threading.current_thread().name, route))
        return {
            "storage_change": 7.0,
            "outward_boundary_flux": 2.0,
            "sources": 3.0,
            "reflux": 1.0,
            "projection": 0.5,
        }


def test_selected_native_balance_forwards_exact_owner_coordinates():
    class _SelectedExecutor:
        def __init__(self):
            self.call = None

        def _selected_accepted_balance_terms(
            self, route, block, component, levels, automatic_terms
        ):
            self.call = (route, block, component, levels, automatic_terms)
            return {
                "storage_change": 7.0,
                "outward_boundary_flux": 2.0,
                "sources": 3.0,
                "reflux": 1.0,
                "projection": 0.5,
            }

    executor = _SelectedExecutor()
    terms = RuntimeConsumerPublisher._native_balance_terms(
        executor,
        "route",
        block="fluid",
        component=2,
        levels=(0, 1),
        automatic_terms=("projection", "reflux"),
    )

    assert executor.call == (
        "route",
        "fluid",
        2,
        [0, 1],
        ["projection", "reflux"],
    )
    assert terms.residual == pytest.approx(4.5)


def _async_balance_runtime(tmp_path: Path):
    base = _install()
    mode = _scientific_output_mode(base.artifact)
    layout = base.artifact.layout_plan.layouts[0]
    block_subject = next(
        assignment.subject
        for assignment in base.artifact.layout_plan.assignments
        if assignment.subject_kind == "block"
    )
    block = BlockHandle(
        block_subject.local_id,
        owner=block_subject.owner_path,
        model_owner=OwnerPath.model("adc-686-balance-fixture"),
    )
    state = Handle(
        "rho",
        kind="state",
        owner=block.owner_path.child(OwnerKind.BLOCK, block.local_id),
    )
    clock = Clock("detached-async-balance", owner=OwnerPath.consumer("adc-686"))
    schedule = every(1, clock=clock)
    ledger = BalanceLedger("detached-async-balance")
    balance = Balance(ledger, block=block, cadence=schedule)
    format_provider = _CapturingFormat(mode)
    descriptor = AsyncScientificOutput(
        format=format_provider,
        schedule=schedule,
        diagnostics=(balance,),
        target="detached-balance",
    )
    node = descriptor.consumer_authoring()[0]
    consumer = Handle("detached-balance", kind="consumer", owner=OwnerPath.consumer("adc-686"))
    diagnostic = DiagnosticQuantity(
        Handle(
            "balance",
            kind="diagnostic",
            owner=consumer.owner_path.child(
                OwnerKind.DESCRIPTOR, consumer.local_id
            ).child(OwnerKind.DESCRIPTOR, "diagnostics"),
        ),
        state,
        "state:fluid",
        layout.handle.qualified_id,
        (0,),
        node.diagnostics[0].diagnostic_execution(),
    )
    manifest = ConsumerManifest(
        consumer,
        ConsumerKind.MONITOR,
        (),
        schedule,
        "detached-balance",
        None,
        mode,
        operation=node.operation,
        diagnostics=node.diagnostics,
        diagnostic_quantities=(diagnostic,),
    )
    graph = ConsumerGraph((manifest,))
    record = replace(
        base.artifact.plan,
        consumer_graph=graph,
        restart_authority=RestartAuthority.from_consumer_graph(graph),
    )
    artifact = CompiledSimulationArtifact(
        record,
        base.artifact.program,
        base.artifact.blocks,
    )
    inputs = BindInputs()
    plan = InstallPlan(
        artifact=artifact,
        bind_inputs=inputs,
        instances={
            installed.name: {"model": installed.model, "spatial": installed.spatial}
            for installed in artifact.blocks
        },
        params=artifact.bind_schema.resolve_bind(
            {}, compile_values=artifact.plan.compile_values
        ),
        aux={},
        execution_context=artifact_execution_context(artifact),
    )
    executor = _BalanceExecutor(plan)
    runtime = RuntimeInstance(plan, executor=executor)
    return runtime, executor, format_provider, manifest, ledger.route_identity(block).token


def test_async_worker_receives_detached_balance_payload_without_reopening_mailbox(tmp_path):
    runtime, executor, format_provider, manifest, route = _async_balance_runtime(tmp_path)
    reports = []
    failures = []

    def run():
        try:
            reports.append(runtime._run(t_end=1.0, max_steps=1, output_dir=tmp_path))
        except BaseException as error:  # noqa: BLE001 - report worker/run failures together
            failures.append(error)

    runner = threading.Thread(target=run, name="adc686-balance-runner", daemon=False)
    runner.start()
    assert format_provider.writer_started.wait(timeout=5)
    assert executor.mailbox_calls == [("adc686-balance-runner", route)]

    executor.mailbox_open = False
    format_provider.release_writer.set()
    runner.join(timeout=10)

    assert not runner.is_alive()
    assert failures == []
    assert len(reports) == 1 and reports[0].accepted_steps == 1
    assert len(format_provider.snapshots) == 1
    assert len(format_provider.worker_threads) == 1
    assert format_provider.worker_threads[0] != "adc686-balance-runner"
    (payload,) = format_provider.snapshots[0].diagnostics
    assert payload.value == pytest.approx(4.5)
    assert dict(payload.terms) == {
        "storage_change": 7.0,
        "outward_boundary_flux": 2.0,
        "sources": 3.0,
        "reflux": 1.0,
        "projection": 0.5,
    }
    assert executor.mailbox_calls == [("adc686-balance-runner", route)]
    accepted = runtime.inspect().to_dict()["instance"]["accepted_diagnostics"]
    assert len(accepted) == 1
    assert accepted[0]["value"] == (4.5).hex()
    assert runtime.consumer_cursors.for_consumer(manifest.qualified_id).committed_samples == 1
