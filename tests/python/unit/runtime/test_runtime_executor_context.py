"""Native runtime resources fail closed before any System constructor is reachable."""

from __future__ import annotations

from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest

from pops import _platform_contracts as platform_contracts
from pops._platform_contracts import (
    ExecutionContext,
    ExecutionResource,
    proven_serial_manifest,
)
from pops.codegen._compiled_artifact import CompiledSimulationArtifact
from pops.codegen._plans import BindInputs, InstallPlan
from pops.identity import make_identity
from pops.model import Handle, OwnerKind, OwnerPath
from pops.output._console_monitor import ConsolePresentation
from pops.output._consumer_contracts import (
    ConsumerGraph,
    ConsumerKind,
    ConsumerManifest,
    ConsumerQuantity,
    ParallelMode,
)
from pops.output._restart_provider import RestartAuthority
from pops.runtime import _multi_layout_executor as multi_executor
from pops.runtime import _platform_manifest as platform_manifest
from pops.runtime import _runtime_executor as executor
from pops.runtime import _runtime_planning as runtime_planning
from pops.runtime._runtime_component_manifests import component_manifests_for_install
from pops.runtime._runtime_plan_contracts import (
    BufferAllocation,
    ClockJoin,
    Collective,
    DeterminismGuarantee,
    Fence,
    RuntimePlanningError,
)
from pops.runtime._runtime_planning import build_runtime_plans
from pops.time import AcceptedStep, Clock, Every, Schedule
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.unit.runtime.test_runtime_planning import _install, _manifest


def _context(*, datatype="float64"):
    backend = proven_serial_manifest(
        backend="production", target="system", abi="test|clang++|c++23", runtime=True
    )
    return ExecutionContext(
        backend=backend,
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", datatype),
        device=ExecutionResource("device", "host"),
    )


@pytest.mark.parametrize(
    "context,match",
    [
        (_context(datatype="float32"), "exact float64"),
    ],
)
def test_unsupported_execution_context_resources_are_rejected(context, match):
    with pytest.raises(NotImplementedError, match=match):
        executor._require_supported_execution_context(
            type(
                "Plan",
                (),
                {
                    "execution_context": context,
                },
            )()
        )


@pytest.mark.parametrize(
    "fact_overrides,error,match",
    [
        (
            {"invert_mpi_active": True},
            NotImplementedError,
            r"requires.*MPI",
        ),
        (
            {
                "mpi_active": False,
                "kokkos_backend": "Cuda",
                "kokkos_device": "cuda",
                "field_memory_space": "managed",
                "kokkos_shared_space": "CudaUVMSpace",
                "kokkos_stream": "cuda::default",
            },
            ValueError,
            "device differs",
        ),
    ],
)
def test_mismatched_native_state_is_rejected_before_system_constructor(
    monkeypatch, fact_overrides, error, match
):
    calls = []

    def forbidden_constructor(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("System constructor became reachable")

    plan = _install()
    context = plan.execution_context
    backend = context.backend
    memory_spaces = backend.memory_spaces.require("runtime.memory_spaces")
    assert len(memory_spaces) == 1
    facts = {
        "mpi_active": False,
        "mpi_ranks": 1,
        "kokkos_backend": backend.capabilities["execution_backend"].require(
            "runtime.execution_backend"
        ),
        "kokkos_device": context.device.identity,
        "field_memory_space": memory_spaces[0],
        "kokkos_shared_space": backend.capabilities["shared_space"].require("runtime.shared_space"),
        "kokkos_stream": backend.capabilities["stream_identity"].require("runtime.stream_identity"),
    }
    overrides = dict(fact_overrides)
    if overrides.pop("invert_mpi_active", False):
        facts["mpi_active"] = context.communicator.identity == "serial"
    facts.update(overrides)
    monkeypatch.setattr(executor, "_native_runtime_facts", lambda: facts)
    # This unit isolates the executor's native-state preflight.  The planning layer is covered
    # separately and an installed MPI/OpenMP wheel must not make this serial fixture fail before the
    # executor is reached.
    monkeypatch.setattr(platform_contracts, "validate_launch", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        platform_contracts, "validate_component_launch", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        platform_manifest, "validate_native_device_resource", lambda *args, **kwargs: None
    )
    monkeypatch.setitem(
        sys.modules, "pops.runtime._system", SimpleNamespace(System=forbidden_constructor)
    )
    runtime_plan = build_runtime_plans(plan, {"fluid": _manifest("fluid")})
    with pytest.raises(error, match=match):
        executor.install_runtime_executor(plan, runtime_plan)
    assert calls == []


def test_runtime_plan_is_required_before_native_preflight(monkeypatch):
    plan = _install()
    calls = []

    def forbidden_preflight(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("native preflight became reachable")

    monkeypatch.setattr(executor, "_require_supported_execution_context", forbidden_preflight)
    with pytest.raises(TypeError, match="exact RuntimePlanBundle"):
        executor.install_runtime_executor(plan)
    assert calls == []


def test_determinism_assumptions_are_rechecked_before_native_preflight(monkeypatch):
    plan = SimpleNamespace(execution_context=SimpleNamespace())
    runtime_plan = SimpleNamespace(
        determinism=DeterminismGuarantee(
            "reproducible",
            ("rank_count",),
            {"rank_count": 1},
            {},
            make_identity("execution-context", {"test": "runtime-executor"}),
        ),
        resources=SimpleNamespace(buffers=()),
        communication=SimpleNamespace(
            collectives=(),
            fences=(),
            clock_joins=(),
        ),
    )
    calls = []

    def forbidden_preflight(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("native preflight became reachable")

    monkeypatch.setattr(executor, "require_install_plan", lambda value: value)
    monkeypatch.setattr(executor, "_require_supported_execution_context", forbidden_preflight)
    monkeypatch.setattr(
        runtime_planning,
        "require_runtime_plan_bundle",
        lambda _plan, value: value,
    )
    monkeypatch.setattr(
        executor,
        "_native_runtime_facts",
        lambda: {
            "mpi_ranks": 2,
        },
    )
    with pytest.raises(RuntimePlanningError) as error:
        executor.install_runtime_executor(plan, runtime_plan)
    assert error.value.code == "determinism_assumption_mismatch"
    assert calls == []


def test_matching_runtime_determinism_assumptions_are_consumed():
    guarantee = DeterminismGuarantee(
        "reproducible",
        ("rank_count",),
        {"rank_count": 1},
        {},
        make_identity("execution-context", {"test": "matching-runtime-executor"}),
    )
    executor._require_runtime_determinism(
        SimpleNamespace(execution_context=SimpleNamespace()),
        SimpleNamespace(
            determinism=guarantee,
            communication=SimpleNamespace(collectives=()),
        ),
        {"mpi_ranks": 1},
    )


def _single_layout_projection():
    layout = SimpleNamespace(handle=SimpleNamespace(qualified_id="layout::primary"))
    call_identity = SimpleNamespace(token="runtime-call::fluid")
    compiled_block = SimpleNamespace(
        name="fluid",
        spatial={"ghost_depth": 2},
    )
    plan = SimpleNamespace(
        artifact=SimpleNamespace(
            blocks=(compiled_block,),
            plan=SimpleNamespace(
                blocks=(compiled_block,),
                field_plans={},
            ),
            layout_plan=SimpleNamespace(
                layouts=(layout,),
                assignments=(
                    SimpleNamespace(
                        subject_kind="block",
                        subject_id="block::fluid",
                        subject=SimpleNamespace(local_id="fluid", qualified_id="block::fluid"),
                        layout=layout.handle,
                    ),
                ),
            ),
        )
    )
    runtime_plan = SimpleNamespace(
        calls=(
            SimpleNamespace(
                identity=call_identity,
                block_id="block::fluid",
                layout_id="layout::primary",
                reads=(SimpleNamespace(resource="state:u"),),
            ),
        ),
        communication=SimpleNamespace(
            transfers=(),
            halos=(
                SimpleNamespace(
                    call_id=call_identity.token,
                    resource="state:u",
                    layout_id="layout::primary",
                    depth=2,
                ),
            ),
        ),
        resources=SimpleNamespace(mapping_provider_ids=(), buffers=()),
    )
    return plan, runtime_plan


def test_single_layout_provider_consumes_exact_call_and_halo_projection():
    plan, runtime_plan = _single_layout_projection()

    executor._require_single_layout_runtime_plan(plan, runtime_plan)

    runtime_plan.calls[0].layout_id = "layout::other"
    with pytest.raises(ValueError, match="calls differ"):
        executor._require_single_layout_runtime_plan(plan, runtime_plan)
    runtime_plan.calls[0].layout_id = "layout::primary"

    runtime_plan.communication.halos[0].layout_id = "layout::other"
    with pytest.raises(ValueError, match="halo differs"):
        executor._require_single_layout_runtime_plan(plan, runtime_plan)


def test_single_layout_provider_authenticates_exact_halo_owner_and_storage():
    plan = _install()
    runtime_plan = build_runtime_plans(
        plan,
        {
            "fluid": _manifest(
                "fluid",
                reads=({"resource": "state:u"},),
                requirements=(
                    {"capability": "halo", "depth": 2, "resource": "state:u"},
                ),
            ),
        },
    )

    executor._require_single_layout_runtime_plan(plan, runtime_plan)


def test_single_layout_provider_refuses_halo_deeper_than_compiled_storage():
    plan = _install()
    runtime_plan = build_runtime_plans(
        plan,
        {
            "fluid": _manifest(
                "fluid",
                reads=({"resource": "state:u"},),
                requirements=(
                    {"capability": "halo", "depth": 3, "resource": "state:u"},
                ),
            ),
        },
    )

    with pytest.raises(ValueError, match="exceeds compiled block.*ghost depth 2"):
        executor._require_single_layout_runtime_plan(plan, runtime_plan)


def test_single_layout_provider_requires_exact_compiled_halo_evidence():
    plan, runtime_plan = _single_layout_projection()
    plan.artifact.blocks[0].spatial = {}

    with pytest.raises(ValueError, match="no exact ghost depth.*numerics/spatial plan"):
        executor._require_single_layout_runtime_plan(plan, runtime_plan)


def test_single_layout_halo_uses_field_plan_depth_above_spatial_depth():
    plan, runtime_plan = _single_layout_projection()
    plan.artifact.plan.field_plans = {
        "potential": SimpleNamespace(
            native_options={
                "output_route": {"owner_block": "fluid"},
                "method": {"ghost_depth": 3},
            },
        ),
    }
    runtime_plan.communication.halos[0].depth = 3

    executor._require_single_layout_runtime_plan(plan, runtime_plan)


def test_single_layout_provider_refuses_halo_without_authenticated_read_owner():
    plan, runtime_plan = _single_layout_projection()
    runtime_plan.communication.halos[0].resource = "state:foreign"

    with pytest.raises(ValueError, match="no authenticated read owner"):
        executor._require_single_layout_runtime_plan(plan, runtime_plan)


@pytest.mark.parametrize(
    "transfers,providers,match",
    [
        ((object(),), (), "layout Transfers"),
        ((), ("pops://mapping/test",), "mapping providers"),
    ],
)
def test_single_layout_provider_refuses_unconsumed_mapping_routes(transfers, providers, match):
    plan, runtime_plan = _single_layout_projection()
    runtime_plan.communication.transfers = transfers
    runtime_plan.resources.mapping_provider_ids = providers

    with pytest.raises(ValueError, match=match):
        executor._require_single_layout_runtime_plan(plan, runtime_plan)


@pytest.mark.parametrize(
    "provider",
    (executor._UniformNativeProvider(), executor._AdaptiveNativeProvider()),
)
def test_single_layout_providers_refuse_call_mismatch_before_geometry(monkeypatch, provider):
    plan, runtime_plan = _single_layout_projection()
    runtime_plan.calls[0].block_id = "block::other"
    reached = []
    monkeypatch.setattr(executor, "require_install_plan", lambda value: value)
    monkeypatch.setattr(executor, "_require_native_geometry", reached.append)

    with pytest.raises(ValueError, match="calls differ"):
        provider.install(plan, runtime_plan)
    assert reached == []


def _multi_layout_projection():
    primary = SimpleNamespace(qualified_id="layout::primary")
    secondary = SimpleNamespace(qualified_id="layout::secondary")
    blocks = (
        SimpleNamespace(name="fluid"),
        SimpleNamespace(name="solid"),
    )
    assignments = tuple(
        SimpleNamespace(
            subject_kind="block",
            subject_id=block_id,
            subject=SimpleNamespace(local_id=name),
            layout=layout,
        )
        for name, block_id, layout in (
            ("fluid", "block::fluid", primary),
            ("solid", "block::solid", secondary),
        )
    )
    plan = SimpleNamespace(
        artifact=SimpleNamespace(
            blocks=blocks,
            layout_plan=SimpleNamespace(assignments=assignments),
        )
    )
    transfer = SimpleNamespace(provider_id="pops://mapping/primary-secondary")
    runtime_plan = SimpleNamespace(
        calls=tuple(
            SimpleNamespace(block_id=block_id, layout_id=layout.qualified_id)
            for block_id, layout in (
                ("block::fluid", primary),
                ("block::solid", secondary),
            )
        ),
        communication=SimpleNamespace(halos=()),
        resources=SimpleNamespace(mapping_provider_ids=("pops://mapping/primary-secondary",)),
    )
    return plan, runtime_plan, (transfer,)


def test_multi_layout_provider_consumes_exact_call_and_mapping_projection():
    plan, runtime_plan, transfers = _multi_layout_projection()

    multi_executor._require_runtime_plan_projection(plan, runtime_plan, transfers)

    runtime_plan.calls[1].layout_id = "layout::primary"
    with pytest.raises(ValueError, match="calls differ"):
        multi_executor._require_runtime_plan_projection(plan, runtime_plan, transfers)
    runtime_plan.calls[1].layout_id = "layout::secondary"

    runtime_plan.resources.mapping_provider_ids = ("pops://mapping/other",)
    with pytest.raises(ValueError, match="mapping providers differ"):
        multi_executor._require_runtime_plan_projection(plan, runtime_plan, transfers)


def test_multi_layout_provider_refuses_unconsumed_halo_plan():
    plan, runtime_plan, transfers = _multi_layout_projection()
    runtime_plan.communication.halos = (object(),)

    with pytest.raises(NotImplementedError, match="explicit per-layout halo scheduler"):
        multi_executor._require_runtime_plan_projection(plan, runtime_plan, transfers)


def _exact_runtime_plan_with_unowned_action(kind):
    names = ("fluid", "solid") if kind == "fence" else ("fluid",)
    plan = _install(names)
    runtime_plan = build_runtime_plans(
        plan,
        {name: _manifest(name) for name in names},
    )
    communication = runtime_plan.communication
    resources = runtime_plan.resources
    if kind == "buffer":
        resources = replace(
            resources,
            buffers=(BufferAllocation("scratch:flux", "host", 256, 0, 0),),
        )
    elif kind == "fence":
        fence = Fence(
            "state:u",
            runtime_plan.calls[0].identity.token,
            runtime_plan.calls[1].identity.token,
            "host",
            "device",
        )
        communication = replace(communication, fences=(fence,))
        resources = replace(resources, fence_ids=(fence.identity.token,))
    elif kind == "clock_join":
        communication = replace(
            communication,
            clock_joins=(
                ClockJoin(
                    runtime_plan.calls[0].identity.token,
                    "fast",
                    "solution",
                    "exact",
                ),
            ),
        )
    else:
        raise AssertionError("unsupported test action")
    return plan, replace(
        runtime_plan,
        communication=communication,
        resources=resources,
    )


@pytest.mark.parametrize(
    "kind,match",
    [
        ("buffer", "buffer allocations"),
        ("fence", "cross-memory fences"),
        ("clock_join", "clock joins"),
    ],
)
def test_unowned_exact_runtime_action_fails_before_native_fact_probe(
    monkeypatch, kind, match
):
    plan, runtime_plan = _exact_runtime_plan_with_unowned_action(kind)

    def forbidden_native_facts():
        raise AssertionError("native fact probe became reachable")

    monkeypatch.setattr(executor, "_native_runtime_facts", forbidden_native_facts)
    with pytest.raises(NotImplementedError, match=match):
        executor.install_runtime_executor(plan, runtime_plan)


def _collective_component_manifest(
    name="fluid",
    *,
    resource="state:u",
    operation="gather",
    strategy="explicit_communicator",
):
    return _manifest(
        name,
        reads=({"resource": resource},),
        requirements=(
            {
                "capability": "collective",
                "resource": resource,
                "operation": operation,
                "strategy": strategy,
            },
        ),
    )


def _collective_owner_graph(plan, *, block_name="fluid"):
    case_owner = plan.artifact.layout_plan.owner
    graph_owner = case_owner.child(OwnerKind.CONSUMER, "graph")
    assignment = next(
        row
        for row in plan.artifact.layout_plan.assignments
        if row.subject_kind == "block" and row.subject.local_id == block_name
    )
    block = assignment.subject
    declaration_owner = OwnerPath.model("runtime-collective-owner")
    declaration = Handle("state", kind="state", owner=declaration_owner)
    reference = declaration._with_owner(
        block.owner_path.child(OwnerKind.BLOCK, block.local_id).instance_of(
            declaration_owner
        ),
        declaration_ref=declaration,
        block_ref=block,
    )
    quantity = ConsumerQuantity(
        reference,
        "state:u",
        assignment.layout.qualified_id,
    )
    clock = Clock("solution", owner=graph_owner)
    return ConsumerGraph((
        ConsumerManifest(
            Handle("console", kind="consumer", owner=graph_owner),
            ConsumerKind.DIAGNOSTIC,
            (quantity,),
            Schedule(Every(AcceptedStep(clock), 1)),
            "console/runtime-collective-owner",
            None,
            ParallelMode.ROOT,
            operation=ConsolePresentation(template=None, handler=None),
        ),
    ))


def _install_with_consumer_graph(base, graph):
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
    return InstallPlan(
        artifact=artifact,
        bind_inputs=BindInputs(),
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


def test_owned_exact_consumer_collective_reaches_native_fact_probe(monkeypatch):
    base = _install()
    plan = _install_with_consumer_graph(base, _collective_owner_graph(base))
    runtime_plan = build_runtime_plans(
        plan,
        component_manifests_for_install(plan),
    )
    assert runtime_plan.communication.collectives[0].strategy == "explicit_communicator"

    class NativeFactProbeReached(Exception):
        pass

    def reached_native_facts():
        raise NativeFactProbeReached

    monkeypatch.setattr(executor, "_native_runtime_facts", reached_native_facts)
    with pytest.raises(NativeFactProbeReached):
        executor.install_runtime_executor(plan, runtime_plan)


def test_unowned_exact_consumer_collective_fails_before_native_fact_probe(monkeypatch):
    plan = _install()
    runtime_plan = build_runtime_plans(
        plan,
        {"fluid": _collective_component_manifest()},
    )

    def forbidden_native_facts():
        raise AssertionError("native fact probe became reachable")

    monkeypatch.setattr(executor, "_native_runtime_facts", forbidden_native_facts)
    with pytest.raises(RuntimePlanningError) as error:
        executor.install_runtime_executor(plan, runtime_plan)
    assert error.value.code == "runtime_collective_without_consumer_owner"


@pytest.mark.parametrize(
    "resource,operation",
    [
        ("state:u", "sum"),
        ("state:foreign", "gather"),
    ],
)
def test_consumer_collective_requires_exact_resource_and_operation_owner(
    monkeypatch, resource, operation
):
    base = _install()
    plan = _install_with_consumer_graph(base, _collective_owner_graph(base))
    runtime_plan = build_runtime_plans(
        plan,
        {
            "fluid": _collective_component_manifest(
                resource=resource,
                operation=operation,
            ),
        },
    )

    def forbidden_native_facts():
        raise AssertionError("native fact probe became reachable")

    monkeypatch.setattr(executor, "_native_runtime_facts", forbidden_native_facts)
    with pytest.raises(RuntimePlanningError) as error:
        executor.install_runtime_executor(plan, runtime_plan)
    assert error.value.code == "runtime_collective_without_consumer_owner"


def test_consumer_collective_requires_exact_block_owner_before_native_fact_probe(
    monkeypatch,
):
    base = _install(("first", "second"))
    plan = _install_with_consumer_graph(
        base,
        _collective_owner_graph(base, block_name="first"),
    )
    runtime_plan = build_runtime_plans(
        plan,
        {
            "first": _manifest("first", reads=({"resource": "state:u"},)),
            "second": _collective_component_manifest("second"),
        },
    )

    def forbidden_native_facts():
        raise AssertionError("native fact probe became reachable")

    monkeypatch.setattr(executor, "_native_runtime_facts", forbidden_native_facts)
    with pytest.raises(RuntimePlanningError) as error:
        executor.install_runtime_executor(plan, runtime_plan)
    assert error.value.code == "runtime_collective_without_consumer_owner"


def test_consumer_collective_requires_exact_strategy_before_native_fact_probe(
    monkeypatch,
):
    base = _install()
    plan = _install_with_consumer_graph(base, _collective_owner_graph(base))
    runtime_plan = build_runtime_plans(
        plan,
        {"fluid": _collective_component_manifest(strategy="ordered")},
    )

    def forbidden_native_facts():
        raise AssertionError("native fact probe became reachable")

    monkeypatch.setattr(executor, "_native_runtime_facts", forbidden_native_facts)
    with pytest.raises(RuntimePlanningError) as error:
        executor.install_runtime_executor(plan, runtime_plan)
    assert error.value.code == "runtime_collective_without_consumer_owner"


def test_consumer_collective_requires_exact_call_requirement_before_native_facts(
    monkeypatch,
):
    base = _install()
    plan = _install_with_consumer_graph(base, _collective_owner_graph(base))
    runtime_plan = build_runtime_plans(
        plan,
        {"fluid": _manifest("fluid", reads=({"resource": "state:u"},))},
    )
    call = runtime_plan.calls[0]
    communicator = runtime_plan.communication.communicator_id
    communication = replace(
        runtime_plan.communication,
        collectives=(
            Collective(
                call.identity.token,
                "state:u",
                "gather",
                "explicit_communicator",
                communicator,
                0,
            ),
        ),
    )
    runtime_plan = replace(runtime_plan, communication=communication)

    def forbidden_native_facts():
        raise AssertionError("native fact probe became reachable")

    monkeypatch.setattr(executor, "_native_runtime_facts", forbidden_native_facts)
    with pytest.raises(RuntimePlanningError) as error:
        executor.install_runtime_executor(plan, runtime_plan)
    assert error.value.code == "runtime_collective_without_consumer_owner"


@pytest.mark.parametrize("replace_plan_communicator", [False, True])
def test_consumer_collective_requires_exact_communicator_before_native_fact_probe(
    monkeypatch, replace_plan_communicator
):
    base = _install()
    plan = _install_with_consumer_graph(base, _collective_owner_graph(base))
    runtime_plan = build_runtime_plans(
        plan,
        {"fluid": _collective_component_manifest()},
    )
    collective = runtime_plan.communication.collectives[0]
    wrong_communicator = collective.communicator_id + ":foreign"
    communication = replace(
        runtime_plan.communication,
        communicator_id=(
            wrong_communicator
            if replace_plan_communicator
            else runtime_plan.communication.communicator_id
        ),
        collectives=(
            Collective(
                collective.call_id,
                collective.resource,
                collective.operation,
                collective.strategy,
                wrong_communicator,
                collective.sequence,
            ),
        ),
    )
    runtime_plan = replace(runtime_plan, communication=communication)

    def forbidden_native_facts():
        raise AssertionError("native fact probe became reachable")

    monkeypatch.setattr(executor, "_native_runtime_facts", forbidden_native_facts)
    with pytest.raises(RuntimePlanningError) as error:
        executor.install_runtime_executor(plan, runtime_plan)
    assert error.value.code == "runtime_collective_without_consumer_owner"


def test_before_step_transfer_cycle_captures_every_native_source_before_any_apply():
    first = SimpleNamespace(mapping_id="A-to-B", source="A", target="B")
    second = SimpleNamespace(mapping_id="B-to-A", source="B", target="A")
    states = {"A": 1, "B": 2}
    events = []

    class Session:
        def __init__(self, transfer):
            self.transfer = transfer
            self.snapshot = None

        def capture(self, generation, attempt):
            events.append(("capture", self.transfer.mapping_id, generation, attempt))
            self.snapshot = states[self.transfer.source]

        def apply(self, generation, attempt):
            events.append(("apply", self.transfer.mapping_id, generation, attempt))
            states[self.transfer.target] = self.snapshot
            return object()

    class NativeEngine:
        def time(self):
            return 0.0

        def macro_step(self):
            return 0

        def step(self, _dt):
            return None

    native = object.__new__(multi_executor._MultiLayoutUniformExecutor)
    native._active_transfer_generation = 7
    native._transfer_attempt = 0
    native._transfer_routes = tuple(
        multi_executor._NativeTransferRoute(
            transfer=row,
            source_block=row.source,
            target_block=row.target,
            session=Session(row),
            source_element_count=1,
            destination_element_count=1,
        )
        for row in (first, second)
    )
    native._engines = {"A": NativeEngine(), "B": NativeEngine()}
    native._mapping_evaluations = {"A-to-B": 0, "B-to-A": 0}
    native._authenticate_mapping_receipt = lambda *args, **kwargs: None
    native._last_mapping_receipts = ()
    native._common_clock = lambda _method: 0

    native.step(0.125)

    assert events == [
        ("capture", "A-to-B", 7, 1),
        ("capture", "B-to-A", 7, 1),
        ("apply", "A-to-B", 7, 1),
        ("apply", "B-to-A", 7, 1),
    ]
    assert states == {"A": 2, "B": 1}
    assert native.mapping_report() == {"A-to-B": 1, "B-to-A": 1}


def test_rejected_multi_layout_attempt_restores_every_child_then_recaptures():
    from pops._bootstrap import StepAttemptRejected

    states = {"A": 1, "B": 2}
    events = []
    first = SimpleNamespace(mapping_id="A-to-B", source="A", target="B")
    second = SimpleNamespace(mapping_id="B-to-A", source="B", target="A")

    class Session:
        def __init__(self, transfer):
            self.transfer = transfer
            self.snapshot = None

        def begin_transaction(self, generation):
            events.append(("begin", self.transfer.mapping_id, generation))

        def capture(self, generation, attempt):
            events.append(("capture", self.transfer.mapping_id, attempt))
            self.snapshot = states[self.transfer.source]

        def apply(self, generation, attempt):
            events.append(("apply", self.transfer.mapping_id, attempt))
            states[self.transfer.target] = self.snapshot
            return object()

        def reject_attempt(self, generation, attempt):
            events.append(("reject", self.transfer.mapping_id, attempt))
            self.snapshot = None

        def rollback_transaction(self, generation):
            events.append(("rollback", self.transfer.mapping_id, generation))

    class Engine:
        def __init__(self, name, reject_once=False):
            self.name = name
            self.reject_once = reject_once
            self.snapshot = None
            self.clock = 0

        def _begin_step_transaction(self):
            self.snapshot = (states[self.name], self.clock)

        def _rollback_step_transaction(self):
            states[self.name], self.clock = self.snapshot
            self.snapshot = None

        def time(self):
            return float(self.clock)

        def macro_step(self):
            return self.clock

        def step(self, _dt):
            if self.reject_once:
                self.reject_once = False
                raise StepAttemptRejected("injected native rejection")
            states[self.name] += 10
            self.clock += 1

    native = object.__new__(multi_executor._MultiLayoutUniformExecutor)
    native._engines = {"A": Engine("A"), "B": Engine("B", reject_once=True)}
    native._transfer_routes = tuple(
        multi_executor._NativeTransferRoute(
            transfer=row,
            source_block=row.source,
            target_block=row.target,
            session=Session(row),
            source_element_count=1,
            destination_element_count=1,
        )
        for row in (first, second)
    )
    native._mapping_evaluations = {"A-to-B": 0, "B-to-A": 0}
    native._mapping_snapshot = None
    native._transfer_generation = 0
    native._active_transfer_generation = None
    native._transfer_attempt = 0
    native._last_mapping_receipts = ()
    native._authenticate_mapping_receipt = lambda *args, **kwargs: None
    native._synchronize_child_temporal_states = lambda: None

    native._begin_step_transaction()
    with pytest.raises(StepAttemptRejected, match="injected native rejection"):
        native.step(0.125)
    assert states == {"A": 1, "B": 2}
    native.step(0.0625)
    assert states == {"A": 12, "B": 11}
    assert native.mapping_report() == {"A-to-B": 1, "B-to-A": 1}
    assert [event for event in events if event[0] == "capture"] == [
        ("capture", "A-to-B", 1),
        ("capture", "B-to-A", 1),
        ("capture", "A-to-B", 2),
        ("capture", "B-to-A", 2),
    ]
    native._rollback_step_transaction()
    assert states == {"A": 1, "B": 2}
    assert native.mapping_report() == {"A-to-B": 0, "B-to-A": 0}


def test_runtime_install_rejects_concurrent_overwrite_transfer_targets():
    common = {
        "operation_abi": 1,
        "target_layout_id": "layout-C",
        "target_subject_id": "state-C",
        "synchronization_uri": "pops://synchronization/before-step@1",
    }
    transfers = (
        SimpleNamespace(mapping_id="A-to-C", **common),
        SimpleNamespace(mapping_id="B-to-C", **common),
    )

    with pytest.raises(ValueError, match="explicit merge protocol"):
        multi_executor._require_unique_transfer_targets(transfers)


def test_cartesian_grid_lowering_is_exact_and_refuses_unrepresentable_geometry():
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.mesh import normalize_layout_plan
    from pops.model import OwnerPath
    from pops.runtime._runtime_mesh_lowering import _uniform_system_values

    def native(grid, name):
        (normalized,) = normalize_layout_plan(Uniform(grid), owner=OwnerPath.case(name)).layouts
        assert normalized.native_spatial_layout is not None
        return normalized.native_spatial_layout

    with pytest.raises(TypeError, match="exact NativeSpatialLayout"):
        _uniform_system_values(SimpleNamespace(n=16, L=2.0, periodic=False))

    square = CartesianGrid(
        frame=Rectangle("square", (0.0, 0.0), (2.0, 2.0)).frame(Cartesian2D()),
        cells=(16, 16),
    )
    assert _uniform_system_values(native(square, "square")) == (
        (16, 16),
        (0.0, 0.0),
        (2.0, 2.0),
        (False, False),
        (((0, 0), (16, 16)),),
        "pops://coordinates/cartesian-2d@1",
    )

    periodic = CartesianGrid(
        frame=square.frame,
        cells=(16, 16),
        periodic=PeriodicAxes(square.frame.axes),
    )
    assert _uniform_system_values(native(periodic, "periodic"))[3] == (True, True)

    partial = CartesianGrid(
        frame=square.frame,
        cells=(16, 16),
        periodic=PeriodicAxes((square.frame.x,)),
    )
    assert _uniform_system_values(native(partial, "partial"))[3] == (True, False)

    rectangular_cells = CartesianGrid(frame=square.frame, cells=(16, 8))
    rectangular_values = _uniform_system_values(native(rectangular_cells, "rectangular"))
    assert rectangular_values[0] == (16, 8)
    assert rectangular_values[2] == (2.0, 2.0)
    shifted = CartesianGrid(
        frame=Rectangle("shifted", (1.0, 0.0), (3.0, 2.0)).frame(Cartesian2D()),
        cells=(16, 16),
    )
    shifted_values = _uniform_system_values(native(shifted, "shifted"))
    assert shifted_values[1:3] == ((1.0, 0.0), (3.0, 2.0))


@pytest.mark.parametrize(
    ("dimension", "shape", "lower", "upper", "periodicity"),
    (
        (1, (11,), (-2.0,), (3.0,), (True,)),
        (
            3,
            (9, 7, 5),
            (-2.0, 1.0, 4.0),
            (3.0, 8.0, 14.0),
            (True, False, True),
        ),
    ),
)
def test_uniform_native_config_lowering_preserves_exact_rank(
    dimension, shape, lower, upper, periodicity
):
    from pops.mesh import NativeSpatialLayout
    from pops.runtime._runtime_mesh_lowering import _uniform_system_values

    layout = NativeSpatialLayout(
        layout_id="rank-%d" % dimension,
        coordinate_system="pops://coordinates/cartesian-%dd@1" % dimension,
        cell_measure={
            1: "pops://cell-measures/cartesian-length@1",
            3: "pops://cell-measures/cartesian-volume@1",
        }[dimension],
        axis_names=("x", "y", "z")[:dimension],
        shape=shape,
        lower=lower,
        upper=upper,
        periodicity=periodicity,
        centering="cell",
        decomposition={
            "schema_version": 1,
            "kind": "single_box",
            "boxes": ({"lower": (0,) * dimension, "upper_exclusive": shape},),
        },
    )

    assert _uniform_system_values(layout) == (
        shape,
        lower,
        upper,
        periodicity,
        (((0,) * dimension, shape),),
        "pops://coordinates/cartesian-%dd@1" % dimension,
    )


def test_uniform_native_config_lowering_refuses_polar_geometry():
    from pops.layouts import Uniform
    from pops.mesh import PolarMesh, normalize_layout_plan
    from pops.model import OwnerPath
    from pops.runtime._runtime_mesh_lowering import _uniform_system_values

    normalized = normalize_layout_plan(
        Uniform(PolarMesh(0.2, 1.0, 8, 16)),
        owner=OwnerPath.case("polar-lowering-refusal"),
    ).layouts[0]
    assert normalized.native_spatial_layout is not None
    with pytest.raises(NotImplementedError, match=r"SystemConfig<2>.*Cartesian"):
        _uniform_system_values(normalized.native_spatial_layout)


def test_uniform_native_config_lowering_rejects_non_tiling_boxes():
    from pops.mesh import NativeSpatialLayout
    from pops.runtime._runtime_mesh_lowering import _uniform_system_values

    layout = NativeSpatialLayout(
        layout_id="gap",
        coordinate_system="pops://coordinates/cartesian-1d@1",
        cell_measure="pops://cell-measures/cartesian-length@1",
        axis_names=("x",),
        shape=(8,),
        lower=(0.0,),
        upper=(1.0,),
        periodicity=(False,),
        centering="cell",
        decomposition={
            "schema_version": 1,
            "kind": "axis_bands",
            "axis": 0,
            "boxes": ({"lower": (0,), "upper_exclusive": (7,)},),
        },
    )
    with pytest.raises(ValueError, match="does not tile"):
        _uniform_system_values(layout)


def test_conservative_multi_layout_average_requires_one_physical_domain():
    from types import SimpleNamespace

    from pops.runtime._multi_layout_executor import (
        _require_conservative_cell_average_geometry,
    )

    def config(
        shape=(16, 12),
        lower=(-2.0, 3.0),
        upper=(-1.0, 4.0),
        periodicity=(True, False),
        coordinate_system="pops://coordinates/cartesian-2d@1",
    ):
        return SimpleNamespace(
            shape=shape,
            lower=lower,
            upper=upper,
            periodicity=periodicity,
            coordinate_system=coordinate_system,
        )

    fine = config()
    coarse = config(shape=(8, 6))
    _require_conservative_cell_average_geometry(fine, coarse)

    with pytest.raises(ValueError, match="identical physical upper bounds"):
        _require_conservative_cell_average_geometry(
            fine,
            config(shape=(8, 6), upper=(0.0, 4.0)),
        )
    with pytest.raises(ValueError, match="identical physical lower bounds"):
        _require_conservative_cell_average_geometry(
            fine,
            config(shape=(8, 6), lower=(0.0, 3.0)),
        )
    with pytest.raises(ValueError, match="identical boundary topology"):
        _require_conservative_cell_average_geometry(
            fine,
            config(shape=(8, 6), periodicity=(False, True)),
        )
    with pytest.raises(TypeError, match="exact ranked periodicity tuple"):
        _require_conservative_cell_average_geometry(
            fine,
            config(shape=(8, 6), periodicity=True),
        )
    with pytest.raises(ValueError, match="identical coordinate systems"):
        _require_conservative_cell_average_geometry(
            fine,
            config(shape=(8, 6), coordinate_system="pops://coordinates/other-2d@1"),
        )
