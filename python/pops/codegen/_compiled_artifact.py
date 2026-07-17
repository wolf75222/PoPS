"""Private immutable compiled-simulation records produced by the public compile phase."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any

from pops.identity import Identity, make_identity

from ._plans import ResolvedSimulationPlan, _deep_freeze, _evidence


@dataclass(frozen=True, slots=True)
class CompiledPlanBlock:
    """Compiler facts retained for one block after authoring objects are discarded."""

    name: str
    backend: str
    spatial: Any
    state_spaces: tuple[str, ...]
    boundaries: tuple[Any, ...] = ()
    state_identities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("CompiledPlanBlock name must be a non-empty string")
        if not isinstance(self.backend, str) or not self.backend:
            raise TypeError("CompiledPlanBlock backend must be a non-empty string")
        state_spaces = tuple(self.state_spaces)
        if len(state_spaces) != 1 or not isinstance(state_spaces[0], str) \
                or not state_spaces[0]:
            raise TypeError("CompiledPlanBlock requires exactly one named state space")
        object.__setattr__(self, "state_spaces", state_spaces)
        state_identities = tuple(self.state_identities)
        if (len(state_identities) != len(state_spaces)
                or any(not isinstance(identity, str) or not identity
                       for identity in state_identities)
                or len(set(state_identities)) != len(state_identities)):
            raise TypeError(
                "CompiledPlanBlock state_identities must uniquely qualify every state space")
        object.__setattr__(self, "state_identities", state_identities)
        object.__setattr__(self, "spatial", _deep_freeze(self.spatial))
        _evidence(self.spatial, where="CompiledPlanBlock.spatial")
        from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan

        boundaries = tuple(
            row if type(row) is CompiledBoundaryPlan else CompiledBoundaryPlan.from_resolved(row)
            for row in self.boundaries
        )
        for boundary in boundaries:
            runtime_data = getattr(boundary, "runtime_boundary_data", None)
            if type(boundary) is not CompiledBoundaryPlan or not callable(runtime_data):
                raise TypeError(
                    "CompiledPlanBlock boundaries must be detached CompiledBoundaryPlan values"
                )
            _evidence(boundary, where="CompiledPlanBlock.boundaries")
        object.__setattr__(self, "boundaries", boundaries)


@dataclass(frozen=True, slots=True)
class CompiledPlanRecord:
    """Detached data required by bind/install; never retains Model or Program builders."""

    plan_identity: Identity
    snapshot: Any
    target: str
    backend: str
    layout: Any
    layout_plan: Any
    layout_targets: Mapping[str, str]
    bind_schema: Any
    compile_values: Mapping[Any, Any]
    field_plans: Mapping[str, Any]
    requirements: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    lowering_coverage: Any
    blocks: tuple[CompiledPlanBlock, ...]
    time_identity: Any
    consumer_graph: Any = None
    restart_authority: Any = None
    component_contracts: tuple[Any, ...] = ()
    resolved_hierarchy: Any = None
    amr_transfer: Any = None
    initial_condition_plan: Any = None
    bootstrap_plan: Any = None
    amr_execution: Any = None
    amr_providers: Mapping[str, Any] = field(default_factory=dict)
    contract_identity: Identity = field(init=False)

    @classmethod
    def from_resolved(cls, plan: ResolvedSimulationPlan) -> CompiledPlanRecord:
        if type(plan) is not ResolvedSimulationPlan:
            raise TypeError("CompiledPlanRecord requires an exact ResolvedSimulationPlan")
        plan.verify()
        return cls(
            plan_identity=Identity.from_data(plan.plan_identity.to_data()),
            snapshot=plan.snapshot,
            target=plan.target,
            backend=plan.backend,
            layout=plan.layout,
            layout_plan=plan.layout_plan,
            layout_targets=plan.layout_targets,
            bind_schema=plan.bind_schema,
            compile_values=plan.compile_values,
            field_plans=plan.field_plans,
            consumer_graph=plan.consumer_graph,
            restart_authority=plan.restart_authority,
            component_contracts=tuple(
                _deep_freeze(item.to_data()) for item in plan.component_inputs),
            requirements=plan.requirements,
            capabilities=plan.capabilities,
            lowering_coverage=plan.lowering_coverage,
            blocks=tuple(
                CompiledPlanBlock(
                    name=block.name,
                    backend=block.backend,
                    spatial=block.spatial,
                    state_spaces=block.state_spaces,
                    boundaries=(() if block.numerics is None else block.numerics.boundaries),
                    state_identities=block.state_identities)
                for block in plan.blocks
            ),
            time_identity=(
                _evidence(plan.time, where="resolved time")
                if plan.time is not None else None
            ),
            resolved_hierarchy=plan.resolved_hierarchy,
            amr_transfer=plan.amr_transfer,
            initial_condition_plan=plan.initial_condition_plan,
            bootstrap_plan=plan.bootstrap_plan,
            amr_execution=plan.amr_execution,
            amr_providers=plan.amr_providers,
        )

    def __post_init__(self) -> None:
        if type(self.plan_identity) is not Identity \
                or self.plan_identity.domain != "resolved-plan":
            raise TypeError("CompiledPlanRecord requires a resolved-plan Identity")
        if self.target not in ("system", "amr_system"):
            raise ValueError("CompiledPlanRecord has an unsupported target")
        object.__setattr__(self, "layout", _deep_freeze(self.layout))
        from pops.mesh import LayoutPlan
        from pops.codegen.lowering_coverage import LoweringCoverageReport
        if type(self.layout_plan) is not LayoutPlan:
            raise TypeError("CompiledPlanRecord.layout_plan must be an exact LayoutPlan")
        targets = dict(self.layout_targets)
        expected_targets = tuple(row.handle.qualified_id for row in self.layout_plan.layouts)
        if tuple(targets) != expected_targets or any(
                value not in ("system", "amr_system") for value in targets.values()):
            raise ValueError("CompiledPlanRecord has invalid per-layout targets")
        object.__setattr__(self, "layout_targets", _deep_freeze(targets))
        if type(self.lowering_coverage) is not LoweringCoverageReport:
            raise TypeError(
                "CompiledPlanRecord.lowering_coverage must be a LoweringCoverageReport")
        object.__setattr__(self, "compile_values", _deep_freeze(self.compile_values))
        object.__setattr__(self, "field_plans", _deep_freeze(self.field_plans))
        if self.consumer_graph is not None:
            from pops.output import ConsumerGraph

            if type(self.consumer_graph) is not ConsumerGraph:
                raise TypeError(
                    "CompiledPlanRecord.consumer_graph must be an exact ConsumerGraph or None")
        from pops.output._restart_provider import RestartAuthority
        if type(self.restart_authority) is not RestartAuthority:
            raise TypeError(
                "CompiledPlanRecord.restart_authority must be an exact RestartAuthority")
        expected_restart = RestartAuthority.from_consumer_graph(self.consumer_graph)
        if self.restart_authority.identity != expected_restart.identity:
            raise ValueError(
                "CompiledPlanRecord.restart_authority differs from its ConsumerGraph")
        object.__setattr__(self, "requirements", _deep_freeze(self.requirements))
        object.__setattr__(self, "capabilities", _deep_freeze(self.capabilities))
        object.__setattr__(self, "amr_providers", _deep_freeze(self.amr_providers))
        contracts = tuple(_deep_freeze(item) for item in self.component_contracts)
        component_ids = [item.get("component_id") for item in contracts]
        if any(not isinstance(component_id, str) or not component_id
               for component_id in component_ids):
            raise TypeError("CompiledPlanRecord component contracts require component_id")
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("CompiledPlanRecord component contracts contain a duplicate")
        object.__setattr__(self, "component_contracts", contracts)
        blocks = tuple(self.blocks)
        if not blocks or any(type(block) is not CompiledPlanBlock for block in blocks):
            raise TypeError("CompiledPlanRecord blocks must be exact CompiledPlanBlock values")
        object.__setattr__(self, "blocks", blocks)
        authorities = (
            self.resolved_hierarchy,
            self.amr_transfer,
            self.initial_condition_plan,
            self.bootstrap_plan,
            self.amr_execution,
        )
        if any(value is not None for value in authorities):
            if self.target != "amr_system" or any(value is None for value in authorities):
                raise ValueError("CompiledPlanRecord has a partial AMR authority set")
            from pops.mesh._amr import (
                BootstrapPlan,
                InitialConditionPlan,
                ResolvedHierarchy,
            )
            from pops.mesh._amr.transfer import ResolvedAMRTransfer
            from pops.amr import AMRExecution
            expected = (
                ResolvedHierarchy,
                ResolvedAMRTransfer,
                InitialConditionPlan,
                BootstrapPlan,
                AMRExecution,
            )
            if any(
                type(value) is not kind
                for value, kind in zip(authorities, expected, strict=True)
            ):
                raise TypeError("CompiledPlanRecord contains a non-exact AMR authority")
        object.__setattr__(
            self, "contract_identity", make_identity("compiled-plan", self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "resolved_plan_identity": self.plan_identity.to_data(),
            "target": self.target,
            "backend": self.backend,
            "layout": _evidence(self.layout, where="compiled plan layout"),
            "layout_plan": _evidence(
                self.layout_plan, where="compiled plan layout plan"),
            "layout_targets": _evidence(
                self.layout_targets, where="compiled plan layout targets"),
            "bind_schema": _evidence(self.bind_schema, where="compiled plan bind schema"),
            "compile_values": _evidence(
                self.compile_values, where="compiled plan compile values"),
            "field_plans": _evidence(
                self.field_plans, where="compiled plan field plans"),
            "consumer_graph": (
                None if self.consumer_graph is None else self.consumer_graph.to_data()
            ),
            "restart_authority": self.restart_authority.to_data(),
            "requirements": _evidence(
                self.requirements, where="compiled plan requirements"),
            "capabilities": _evidence(
                self.capabilities, where="compiled plan capabilities"),
            "component_contracts": _evidence(
                self.component_contracts, where="compiled plan component contracts"),
            "lowering_coverage": _evidence(
                self.lowering_coverage, where="compiled plan lowering coverage"),
            "blocks": [
                {
                    "name": block.name,
                    "backend": block.backend,
                    "state_spaces": block.state_spaces,
                    "state_identities": block.state_identities,
                    "spatial": _evidence(
                        block.spatial, where="compiled plan block spatial"),
                    "boundaries": _evidence(
                        block.boundaries, where="compiled plan block boundaries"),
                }
                for block in self.blocks
            ],
            "time_identity": self.time_identity,
            "resolved_hierarchy": _evidence(
                self.resolved_hierarchy, where="compiled plan resolved hierarchy"
            ) if self.resolved_hierarchy is not None else None,
            "amr_transfer": _evidence(
                self.amr_transfer, where="compiled plan AMR transfer"
            ) if self.amr_transfer is not None else None,
            "initial_condition_plan": _evidence(
                self.initial_condition_plan, where="compiled plan initial conditions"
            ) if self.initial_condition_plan is not None else None,
            "bootstrap_plan": _evidence(
                self.bootstrap_plan, where="compiled plan bootstrap"
            ) if self.bootstrap_plan is not None else None,
            "amr_execution": _evidence(
                self.amr_execution, where="compiled plan AMR execution"
            ) if self.amr_execution is not None else None,
            "amr_providers": _evidence(
                self.amr_providers, where="compiled plan AMR providers"),
        }

    def verify(self) -> None:
        if self.contract_identity != make_identity("compiled-plan", self._payload()):
            raise ValueError("CompiledPlanRecord identity verification failed")


def _binary_evidence(value: Any, *, where: str) -> dict[str, Any]:
    """Authenticate one compiled component, including current binary bytes when available."""
    path = getattr(value, "so_path", None)
    digest = None
    if isinstance(path, (str, os.PathLike)) and os.path.isfile(path):
        hashed = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hashed.update(chunk)
        digest = hashed.hexdigest()
    metadata = {}
    for name in (
        "target", "backend", "abi_key", "model_hash", "definition_identity",
        "module_manifest",
    ):
        item = getattr(value, name, None)
        if item is not None:
            metadata[name] = _evidence(item, where="%s.%s" % (where, name))
    if hasattr(value, "wave_speed_provider"):
        metadata["wave_speed_provider"] = _evidence(
            value.wave_speed_provider,
            where="%s.wave_speed_provider" % where,
        )
    return {
        "component": _evidence(value, where=where),
        "binary_sha256": digest,
        "metadata": metadata,
    }


def _seal_component(value: Any) -> None:
    seal = getattr(value, "_seal", None)
    if callable(seal) and not getattr(value, "_sealed", False):
        seal()


def _common_platform_manifest(
    *, backend: str, target: str, blocks: tuple[Any, ...], programs: tuple[Any, ...],
    external: tuple[Any, ...],
) -> Any:
    """Prove one platform contract from every executable binary, never a representative."""
    from pops import _pops
    from pops._platform_contracts import artifact_platform_manifest
    from pops.codegen._native_mpi import native_mpi_communicator

    # Compilation selects the communicator seam baked into the host module, independently of world
    # size or whether a report happened to observe an initialized process.  A size-one MPI job still
    # produces an MPI_COMM_WORLD artifact and must not alias a genuinely serial binary.
    communicator = native_mpi_communicator(_pops)
    components = tuple(block.model for block in blocks)
    components += tuple(programs)
    manifests = tuple(
        artifact_platform_manifest(
            backend=backend, target=target, component=component,
            communicator=communicator,
        )
        for component in components
    )
    baseline = manifests[0]
    mismatch = [index for index, manifest in enumerate(manifests[1:], 1)
                if manifest.to_data() != baseline.to_data()]
    if mismatch:
        raise ValueError(
            "compiled executable components do not prove one common PlatformManifest; "
            "mismatching component indices=%s" % mismatch)
    if external:
        for component in external:
            component_communicator = component.platform_manifest.communicator.require(
                "external component communicator")
            if component_communicator != communicator:
                raise ValueError(
                    "external native component communicator differs from the compiled host route: "
                    "expected %r, got %r" % (communicator, component_communicator))
        # The complete ABI/device/precision gate runs against the explicit ExecutionContext in
        # InstallPlan.  Compilation has no authority to fabricate that runtime resource.
    return baseline


@dataclass(frozen=True, slots=True)
class CompiledBlockArtifact:
    """One resolved block paired with its authenticated native component."""

    name: str
    model: Any
    spatial: Any
    state_spaces: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("CompiledBlockArtifact name must be a non-empty string")
        state_spaces = tuple(self.state_spaces)
        if len(state_spaces) != 1 or not isinstance(state_spaces[0], str) \
                or not state_spaces[0]:
            raise TypeError("CompiledBlockArtifact requires exactly one named state space")
        object.__setattr__(self, "state_spaces", state_spaces)
        _binary_evidence(self.model, where="CompiledBlockArtifact.model")
        object.__setattr__(self, "spatial", _deep_freeze(self.spatial))
        _evidence(self.spatial, where="CompiledBlockArtifact.spatial")


@dataclass(frozen=True, slots=True)
class CompiledLayoutProgram:
    """One independently installable Program binary for one authenticated layout partition."""

    layout_id: str
    target: str
    block_names: tuple[str, ...]
    program: Any
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise TypeError("CompiledLayoutProgram.layout_id must be non-empty")
        if self.target not in ("system", "amr_system"):
            raise ValueError("CompiledLayoutProgram target is unsupported")
        names = tuple(self.block_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise TypeError("CompiledLayoutProgram.block_names must contain non-empty names")
        if len(names) != len(set(names)):
            raise ValueError("CompiledLayoutProgram.block_names contains a duplicate")
        object.__setattr__(self, "block_names", names)
        routes = tuple(name for _index, name in getattr(self.program, "program_block_routes", ()))
        if routes != names:
            raise ValueError(
                "CompiledLayoutProgram binary routes do not match its exact block partition")
        _seal_component(self.program)
        object.__setattr__(self, "identity", make_identity("layout-program", self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "layout_id": self.layout_id,
            "target": self.target,
            "block_names": list(self.block_names),
            "binary": _binary_evidence(self.program, where="compiled layout program"),
        }

    def verify(self) -> None:
        if self.identity != make_identity("layout-program", self._payload()):
            raise ValueError("CompiledLayoutProgram identity verification failed")

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.to_data()}


@dataclass(frozen=True, slots=True)
class CompiledSimulationArtifact:
    """The single concrete, immutable output type of the compile phase.

    The runtime-coupled program/model handles remain internal components.  Public inspection is
    delegated through this exact wrapper, while its identity authenticates the resolved plan and
    every binary component.
    """

    plan: Any
    program: Any | None
    blocks: tuple[CompiledBlockArtifact, ...]
    layout_programs: tuple[CompiledLayoutProgram, ...] = ()
    component_artifacts: tuple[Any, ...] = ()
    artifact_identity: Identity = field(init=False)
    platform_manifest: Any = field(init=False)
    _component_evidence: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.plan) is ResolvedSimulationPlan:
            object.__setattr__(self, "plan", CompiledPlanRecord.from_resolved(self.plan))
        if type(self.plan) is not CompiledPlanRecord:
            raise TypeError(
                "CompiledSimulationArtifact.plan must originate from an exact "
                "ResolvedSimulationPlan")
        self.plan.verify()
        blocks = tuple(self.blocks)
        if not blocks or any(type(block) is not CompiledBlockArtifact for block in blocks):
            raise TypeError(
                "CompiledSimulationArtifact.blocks must contain exact CompiledBlockArtifact values")
        expected = tuple(block.name for block in self.plan.blocks)
        actual = tuple(block.name for block in blocks)
        if actual != expected:
            raise ValueError(
                "CompiledSimulationArtifact blocks must match resolved plan order exactly")
        for compiled, resolved in zip(blocks, self.plan.blocks, strict=True):
            if compiled.state_spaces != resolved.state_spaces:
                raise ValueError(
                    "compiled block %r changed the resolved state-space route" % compiled.name)
            if _evidence(compiled.spatial, where="compiled spatial") != _evidence(
                    resolved.spatial, where="resolved spatial"):
                raise ValueError(
                    "compiled block %r changed the resolved spatial descriptor" % compiled.name)
        layout_programs = tuple(self.layout_programs)
        if any(type(row) is not CompiledLayoutProgram for row in layout_programs):
            raise TypeError("layout_programs must contain exact CompiledLayoutProgram values")
        expected_layouts = tuple(row.handle.qualified_id for row in self.plan.layout_plan.layouts)
        assignments = {
            row.subject.local_id: row.layout.qualified_id
            for row in self.plan.layout_plan.assignments if row.subject_kind == "block"
        }
        expected_partitions = {
            layout_id: tuple(block.name for block in blocks
                             if assignments.get(block.name) == layout_id)
            for layout_id in expected_layouts
        }
        if not layout_programs and len(expected_layouts) == 1 and self.program is not None:
            layout_id = expected_layouts[0]
            layout_programs = (CompiledLayoutProgram(
                layout_id, self.plan.layout_targets[layout_id],
                expected_partitions[layout_id], self.program),)
        # Uniform execution always requires one independently installable Program per layout.
        # AMR permits a program-less low-level artifact, but when a compiled Program is present it
        # is an equally authenticated per-layout binary (multi-layout AMR is refused at resolve).
        required_program_layouts = expected_layouts \
            if self.plan.target == "system" or self.program is not None else ()
        if tuple(row.layout_id for row in layout_programs) != required_program_layouts:
            raise ValueError(
                "layout_programs must cover every and only per-layout system target")
        for row in layout_programs:
            if row.target != self.plan.layout_targets[row.layout_id]:
                raise ValueError("layout Program target differs from resolved per-layout target")
            if row.block_names != expected_partitions[row.layout_id]:
                raise ValueError("layout Program block partition differs from LayoutPlan")
            row.verify()
        if len(layout_programs) == 1:
            if self.program is not layout_programs[0].program:
                raise ValueError("single-layout artifact program must be its layout Program")
        elif self.program is not None:
            raise ValueError("multi-layout artifact cannot retain an ambiguous global Program")
        object.__setattr__(self, "layout_programs", layout_programs)
        if self.program is not None:
            _seal_component(self.program)
        for block in blocks:
            _seal_component(block.model)
        for block in blocks:
            target = getattr(block.model, "target", self.plan.target)
            if target != self.plan.target:
                raise ValueError(
                    "compiled block %r target=%r does not match resolved target=%r"
                    % (block.name, target, self.plan.target))
        for compiled, resolved in zip(blocks, self.plan.blocks, strict=True):
            backend = getattr(compiled.model, "backend", None)
            if backend != resolved.backend:
                raise ValueError(
                    "compiled block %r backend=%r does not match resolved backend=%r"
                    % (compiled.name, backend, resolved.backend))
        object.__setattr__(self, "blocks", blocks)
        from pops.external import CompiledComponentArtifact
        component_artifacts = tuple(self.component_artifacts)
        if any(type(item) is not CompiledComponentArtifact for item in component_artifacts):
            raise TypeError(
                "CompiledSimulationArtifact.component_artifacts must contain exact values")
        expected_components = tuple(
            item["component_id"] for item in self.plan.component_contracts)
        actual_components = tuple(item.component_id for item in component_artifacts)
        if actual_components != expected_components:
            raise ValueError(
                "compiled component artifacts must match resolved component order exactly")
        for item in component_artifacts:
            item.verify()
        object.__setattr__(self, "component_artifacts", component_artifacts)
        platform = _common_platform_manifest(
            backend=self.plan.backend, target=self.plan.target, blocks=blocks,
            programs=tuple(row.program for row in layout_programs),
            external=component_artifacts)
        object.__setattr__(self, "platform_manifest", platform)
        evidence = self._current_component_evidence()
        object.__setattr__(self, "_component_evidence", _deep_freeze(evidence))
        object.__setattr__(
            self, "artifact_identity", make_identity(
                "artifact", self._payload(evidence, platform)))

    @property
    def authoring_snapshot(self) -> Any:
        return self.plan.snapshot

    @property
    def semantic_identity(self) -> Identity:
        return self.plan.snapshot.semantic_identity

    @property
    def bind_schema(self) -> Any:
        return self.plan.bind_schema

    @property
    def target(self) -> str:
        return self.plan.target

    @property
    def layout(self) -> Any:
        return self.plan.layout

    @property
    def layout_plan(self) -> Any:
        return self.plan.layout_plan

    @property
    def so_path(self) -> str:
        return str(self._common_executable_attribute("so_path"))

    @property
    def abi_key(self) -> Any:
        return self._common_executable_attribute("abi_key")

    @property
    def cxx(self) -> Any:
        return self._common_executable_attribute("cxx")

    @property
    def std(self) -> Any:
        return self._common_executable_attribute("std")

    @property
    def backend(self) -> str:
        return self.plan.backend

    @property
    def program_name(self) -> Any:
        return getattr(self.program, "program_name", None)

    @property
    def program_hash(self) -> Any:
        return getattr(self.program, "program_hash", None)

    @property
    def cache_key(self) -> Any:
        return self._common_executable_attribute("cache_key", absent=None)

    @property
    def codegen_env(self) -> Any:
        return self._common_executable_attribute("codegen_env", absent=None)

    @property
    def module_manifest(self) -> Any:
        # A Module manifest authenticates the compiled *model*, not the per-layout
        # Program executable.  The latter is the right authority for ``so_path``
        # and ``abi_key``, but a Program compiled from the frozen model graph does
        # not itself carry a second Module manifest.  Report the common block-model
        # manifest when the aggregate has one unambiguous model authority; a
        # heterogeneous multi-model artifact remains explicitly non-scalar.
        manifests = tuple(
            getattr(block.model, "module_manifest", None) for block in self.blocks)
        if not manifests or manifests[0] is None:
            return None
        authority = manifests[0].to_dict()
        if any(manifest is None or manifest.to_dict() != authority
               for manifest in manifests[1:]):
            return None
        return manifests[0]

    @property
    def lowering_coverage(self) -> Any:
        from pops.codegen.lowering_coverage import LoweringCoverageReport, LoweringCoverageRow

        rows = list(self.plan.lowering_coverage.rows)

        def append_component(prefix: str, component: Any) -> None:
            coverage = getattr(component, "lowering_coverage", None)
            if coverage is None:
                coverage = getattr(component, "lowering_coverage_report", None)
            if type(coverage) is not LoweringCoverageReport:
                return
            for row in coverage.rows:
                rows.append(LoweringCoverageRow(
                    source="%s/%s" % (prefix, row.source),
                    disposition=row.disposition,
                    targets=tuple("%s/%s" % (prefix, target) for target in row.targets),
                    rule=row.rule,
                    gate=row.gate,
                ))

        for block in self.blocks:
            append_component("block:%s" % block.name, block.model)
        for row in self.layout_programs:
            append_component("layout-program:%s" % row.layout_id, row.program)
        return LoweringCoverageReport(rows)

    @property
    def program_param_routes(self) -> Any:
        if self.program is None:
            return None
        return getattr(self.program, "program_param_routes", None)

    @property
    def program_block_routes(self) -> tuple[tuple[int, str], ...]:
        if self.program is None:
            return ()
        routes = getattr(self.program, "program_block_routes", None)
        if routes is None:
            raise ValueError("compiled program lacks immutable block-route metadata")
        return tuple(routes)

    def program_for_layout(self, layout_id: str) -> CompiledLayoutProgram:
        matches = [row for row in self.layout_programs if row.layout_id == layout_id]
        if len(matches) != 1:
            raise KeyError("artifact has no exact Program for layout %s" % layout_id)
        return matches[0]

    @property
    def layout_program_paths(self) -> dict[str, str]:
        return {row.layout_id: str(row.program.so_path) for row in self.layout_programs}

    def _executable_components(self) -> tuple[Any, ...]:
        programs = tuple(row.program for row in self.layout_programs)
        return programs if programs else tuple(block.model for block in self.blocks)

    def _common_executable_attribute(self, name: str, *, absent: Any = ...) -> Any:
        values = tuple(getattr(component, name, absent) for component in self._executable_components())
        if absent is ... and any(value is ... for value in values):
            raise AttributeError("compiled executable component lacks %s" % name)
        if any(value != values[0] for value in values[1:]):
            raise ValueError(
                "aggregate artifact has no scalar %s; inspect qualified layout/component evidence"
                % name)
        return values[0]

    def _current_component_evidence(self) -> dict[str, Any]:
        evidence = {
            "blocks": [
                {
                    "name": block.name,
                    "state_spaces": block.state_spaces,
                    "binary": _binary_evidence(
                        block.model, where="artifact.block[%r]" % block.name),
                    "spatial": _evidence(block.spatial, where="artifact.block.spatial"),
                }
                for block in self.blocks
            ],
            "program": (
                _binary_evidence(self.program, where="artifact.program")
                if self.program is not None else None
            ),
            "layout_programs": [row.to_data() for row in self.layout_programs],
            "external": [item.to_data() for item in self.component_artifacts],
        }
        return evidence

    def _payload(self, evidence: Any, platform: Any | None = None) -> dict[str, Any]:
        selected = self.platform_manifest if platform is None else platform
        return {
            "schema_version": 2,
            "plan_identity": self.plan.plan_identity.to_data(),
            "target": self.plan.target,
            "platform_manifest": selected.to_data(),
            "components": evidence,
        }

    def verify(self) -> None:
        self.plan.verify()
        for item in self.component_artifacts:
            item.verify()
        for item in self.layout_programs:
            item.verify()
        current = self._current_component_evidence()
        current_platform = _common_platform_manifest(
            backend=self.plan.backend, target=self.plan.target, blocks=self.blocks,
            programs=tuple(row.program for row in self.layout_programs),
            external=self.component_artifacts)
        expected = make_identity("artifact", self._payload(current, current_platform))
        if (self.artifact_identity != expected
                or self._component_evidence != _deep_freeze(current)
                or self.platform_manifest != current_platform):
            raise ValueError("CompiledSimulationArtifact identity verification failed")

    def inspect(self) -> Any:
        from pops.codegen.inspect_report import build_compiled_report

        return build_compiled_report(self)

    def requirements(self) -> Any:
        from pops.codegen.inspect_report import build_requirements

        return build_requirements(self)

    def manifest(self) -> Any:
        from pops.external.artifact_manifest import build_compiled_manifest

        return build_compiled_manifest(self)

    def arguments(self) -> Any:
        from pops.codegen.inspect_compiled import build_arguments

        return build_arguments(self)

    def estimate_memory(self, mesh: Any, *, platform: Any = None, layout: Any = None) -> Any:
        from pops.codegen.inspect_compiled import build_memory_estimate

        return build_memory_estimate(
            self, mesh, platform=platform, layout=layout or self.layout)

    def scratch_plan(self) -> Any:
        if self.program is None:
            raise ValueError("compiled artifact has no whole-system Program to analyze")
        from pops.codegen.scratch_plan import build_scratch_plan

        return build_scratch_plan(self.program.program)


__all__ = [
    "CompiledBlockArtifact", "CompiledPlanBlock", "CompiledPlanRecord",
    "CompiledSimulationArtifact",
]
