"""ADC-684 phase A: exact immutable plans derived before RuntimeInstance cutover."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace

import pytest

from pops._platform_contracts import (
    CapabilityProof,
    ExecutionContext,
    ExecutionResource,
    proven_serial_manifest,
)
from pops.codegen._plans import BindInputs, InstallPlan, ResolvedBlock, ResolvedSimulationPlan
from pops.codegen.compiled_artifact import CompiledBlockArtifact, CompiledSimulationArtifact
from pops.codegen.lowering_coverage import LoweringCoverageReport
from pops.mesh import CartesianMesh, LayoutPlanBuilder
from pops.mesh.layout_plan import LayoutMappingRequirement
from pops.mesh.layouts import Uniform
from pops.model import ComponentManifest, Handle, OwnerPath
from pops.model.bind_schema import BindSchema
from pops.problem._snapshot import AuthoringSnapshot
from pops.runtime._runtime_plan_contracts import RuntimePlanBundle, RuntimePlanningError
from pops.runtime._runtime_planning import build_runtime_plans
from tests.python.unit.codegen._typed_artifact_fixture import CanonicalValue, CompiledComponent


@dataclass(frozen=True)
class _MappingProvider:
    qualified_id: str
    routes: frozenset[tuple[str, str]]

    def canonical_identity(self):
        return {"qualified_id": self.qualified_id, "routes": sorted(self.routes)}

    def supports_layout_mapping(self, requirement: LayoutMappingRequirement) -> bool:
        return (requirement.source.qualified_id, requirement.target.qualified_id) in self.routes


def _layout(names, *, heterogeneous=False):
    owner = OwnerPath.case("runtime-planning-" + "-".join(names))
    blocks = tuple(Handle(name, kind="block", owner=owner) for name in names)
    builder = LayoutPlanBuilder(owner)
    first = builder.layout("primary", Uniform(CartesianMesh(n=8)))
    second = builder.layout("secondary", Uniform(CartesianMesh(n=16))) if heterogeneous else first
    for index, block in enumerate(blocks):
        builder.assign_block(block, second if heterogeneous and index else first)
    providers = ()
    if heterogeneous:
        (requirement,) = builder.require_mapping(first, second, channel="state")
        providers = (
            _MappingProvider(
                "pops://test/mapping/primary-secondary",
                frozenset(((requirement.source.qualified_id, requirement.target.qualified_id),)),
            ),
        )
    return builder.resolve(blocks=blocks, providers=providers)


def _artifact(names=("fluid",), *, heterogeneous=False, memory_spaces=("host",)) -> CompiledSimulationArtifact:
    layout_plan = _layout(names, heterogeneous=heterogeneous)
    schema = BindSchema()
    resolved = ResolvedSimulationPlan(
        snapshot=AuthoringSnapshot({"case": "runtime-planning", "blocks": list(names)}),
        target="system",
        backend="production",
        layout={"kind": "runtime-planning"},
        layout_plan=layout_plan,
        time=CanonicalValue("rk2"),
        blocks=tuple(
            ResolvedBlock(
                name, CanonicalValue("source-" + name), {"mesh": name},
                "production", ("U",))
            for name in names
        ),
        bind_schema=schema,
        compile_values=schema.resolve_compile(),
        field_plans={},
        libraries=(),
        requirements={},
        capabilities={},
        lowering_coverage=LoweringCoverageReport(()),
    )
    components = tuple(CompiledComponent(name, target="system") for name in names)
    for component in components:
        component.memory_spaces = memory_spaces
    blocks = tuple(
        CompiledBlockArtifact(name, component, planned.spatial, planned.state_spaces)
        for name, component, planned in zip(names, components, resolved.blocks, strict=True)
    )
    return CompiledSimulationArtifact(resolved, CompiledComponent("program", target="system"), blocks)


def _context(artifact: CompiledSimulationArtifact, memory_spaces: tuple[str, ...]):
    if memory_spaces == ("host",):
        return None
    evidence = "test.explicit-memory-spaces.v1"
    proof = CapabilityProof.proven
    backend = proven_serial_manifest(
        backend="production",
        target="system",
        abi=artifact.platform_manifest.abi.require("artifact.abi"),
        runtime=True,
    )
    backend = replace(backend, memory_spaces=proof(memory_spaces, evidence))
    return ExecutionContext(
        backend=backend,
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", "float64"),
        device=ExecutionResource("device", "host"),
    )


def _install(names=("fluid",), *, heterogeneous=False, memory_spaces=("host",)) -> InstallPlan:
    artifact = _artifact(names, heterogeneous=heterogeneous, memory_spaces=memory_spaces)
    inputs = BindInputs()
    return InstallPlan(
        artifact=artifact,
        bind_inputs=inputs,
        instances={block.name: {"model": block.model, "spatial": block.spatial} for block in artifact.blocks},
        params=artifact.bind_schema.resolve_bind({}, compile_values=artifact.plan.compile_values),
        aux={},
        execution_context=_context(artifact, memory_spaces),
    )


def _manifest(
    name,
    *,
    reads=(),
    writes=(),
    requirements=(),
    effects=(),
    clocks=None,
    determinism="reproducible",
    scope=("rank_count",),
):
    return ComponentManifest(
        uri="pops://runtime-planning.test/components/%s" % name,
        component_type="spatial_operator",
        version="1.0.0",
        reads=reads,
        writes=writes,
        requirements=requirements,
        effects=effects,
        clocks=({"clock": "solution", "access": "stage"},) if clocks is None else clocks,
        target={"variants": [{"dimension": 2, "scalar": "float64", "device": "host", "features": []}]},
        determinism={"classification": determinism, "scope": list(scope)},
        precision={"inputs": ["float64"], "accumulation": "float64", "outputs": ["float64"]},
        entry_points={"step": "pops_test_%s_step" % name},
    )


def test_builder_derives_halo_buffers_resources_and_canonical_round_trip():
    install = _install()
    manifest = _manifest(
        "fluid",
        reads=({"resource": "state:u"},),
        writes=({"resource": "rate:u"},),
        requirements=(
            {"capability": "halo", "depth": 1},
            {"capability": "halo", "depth": 3},
            {"capability": "buffer", "resource": "scratch:flux", "bytes": 256},
        ),
        effects=({"kind": "state_read", "resource": "state:u"},),
    )
    before = install.bind_identity

    bundle = build_runtime_plans(install, {"fluid": manifest})

    assert install.bind_identity == before
    assert bundle.calls[0].component_manifest_identity == manifest.semantic_digest
    assert [(row.resource, row.depth) for row in bundle.communication.halos] == [("state:u", 3)]
    assert bundle.resources.buffers[0].to_data() == {
        "resource": "scratch:flux",
        "memory_space": "host",
        "size_bytes": 256,
        "first_call": 0,
        "last_call": 0,
    }
    assert bundle.determinism.assumptions["rank_count"] == 1
    assert RuntimePlanBundle.from_data(bundle.to_data()) == bundle


def test_strict_decoder_rejects_unknown_fields_and_tampered_identity():
    bundle = build_runtime_plans(_install(), {"fluid": _manifest("fluid")})
    unknown = bundle.to_data()
    unknown["legacy"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        RuntimePlanBundle.from_data(unknown)
    tampered = bundle.to_data()
    tampered["layout_plan_id"] += "-changed"
    with pytest.raises(ValueError, match="layout identities"):
        RuntimePlanBundle.from_data(tampered)


def test_values_are_deeply_immutable_and_assumptions_fail_closed():
    bundle = build_runtime_plans(_install(), {"fluid": _manifest("fluid", determinism="bitwise")})
    with pytest.raises(FrozenInstanceError):
        bundle.calls[0].ordinal = 4
    with pytest.raises(TypeError):
        bundle.determinism.assumptions["rank_count"] = 2
    changed = dict(bundle.determinism.assumptions)
    changed["rank_count"] = 2
    with pytest.raises(RuntimePlanningError) as error:
        bundle.determinism.require_assumptions(changed)
    assert error.value.code == "determinism_assumption_mismatch"


def test_heterogeneous_layouts_emit_only_authenticated_directional_transfers():
    install = _install(("fluid", "solid"), heterogeneous=True)
    bundle = build_runtime_plans(install, {"fluid": _manifest("fluid"), "solid": _manifest("solid")})
    transfer = bundle.communication.transfers
    assert len(transfer) == 1
    assert transfer[0].source_layout_id != transfer[0].target_layout_id
    assert transfer[0].provider_id == "pops://test/mapping/primary-secondary"
    assert bundle.resources.mapping_provider_ids == (transfer[0].provider_id,)
    assert bundle.calls[0].component_manifest_identity != bundle.calls[1].component_manifest_identity


def test_device_write_followed_by_host_read_derives_one_fence():
    install = _install(("producer", "consumer"), memory_spaces=("device", "host"))
    manifests = {
        "producer": _manifest("producer", writes=({"resource": "state:u", "memory_space": "device"},)),
        "consumer": _manifest("consumer", reads=({"resource": "state:u", "memory_space": "host"},)),
    }
    bundle = build_runtime_plans(install, manifests)
    assert len(bundle.communication.fences) == 1
    fence = bundle.communication.fences[0]
    assert (fence.resource, fence.source_space, fence.target_space) == ("state:u", "device", "host")
    assert bundle.resources.fence_ids == (fence.identity.token,)


def test_collective_order_and_strategy_enter_bitwise_assumptions():
    install = _install(("first", "second"))
    manifests = {
        name: _manifest(
            name,
            reads=({"resource": "diagnostic:%s" % name},),
            requirements=({"capability": "collective",
                           "resource": "diagnostic:%s" % name,
                           "operation": "sum", "strategy": "ordered_tree"},),
            determinism="bitwise")
        for name in ("first", "second")
    }
    bundle = build_runtime_plans(install, manifests)
    collectives = bundle.communication.collectives
    assert [row.sequence for row in collectives] == [0, 1]
    assert bundle.determinism.assumptions["reduction_order"] == tuple(
        row.identity.token for row in collectives)
    assert bundle.determinism.assumptions["reduction_strategy"] == (
        "sum:ordered_tree", "sum:ordered_tree")


@pytest.mark.parametrize(
    "requirements,code",
    [
        (({"capability": "halo", "depth": 2},), "ambiguous_halo_resource"),
        (({"capability": "magic"},), "unsupported_runtime_requirement"),
        (({"capability": "halo", "depth": 2, "resource": "state:missing"},),
         "halo_without_declared_read"),
    ],
)
def test_unproved_or_ambiguous_runtime_requirements_are_refused(requirements, code):
    manifest = _manifest("fluid", reads=({"resource": "state:u"}, {"resource": "state:v"}), requirements=requirements)
    with pytest.raises(RuntimePlanningError) as error:
        build_runtime_plans(_install(), {"fluid": manifest})
    assert error.value.code == code


def test_distinct_clocks_require_an_explicit_connected_join():
    install = _install(("fast", "slow"))
    manifests = {
        "fast": _manifest("fast", clocks=({"clock": "fast", "access": "stage"},)),
        "slow": _manifest("slow", clocks=({"clock": "slow", "access": "stage"},)),
    }
    with pytest.raises(RuntimePlanningError) as error:
        build_runtime_plans(install, manifests)
    assert error.value.code == "missing_clock_join"

    manifests["fast"] = _manifest(
        "fast",
        clocks=({"clock": "fast", "access": "join", "target": "slow", "policy": "exact_sync"},),
    )
    bundle = build_runtime_plans(install, manifests)
    assert [(row.source_clock, row.target_clock, row.policy) for row in bundle.communication.clock_joins] == [("fast", "slow", "exact_sync")]


def test_component_set_and_unspecified_determinism_are_strict_refusals():
    install = _install()
    with pytest.raises(RuntimePlanningError) as missing:
        build_runtime_plans(install, {})
    assert missing.value.code == "component_set_mismatch"
    with pytest.raises(RuntimePlanningError) as unknown:
        build_runtime_plans(install, {"fluid": _manifest("fluid", determinism="unspecified", scope=())})
    assert unknown.value.code == "unspecified_component_determinism"
