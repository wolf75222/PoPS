"""Concrete immutable compiled-simulation artifact returned by the public compile phase."""
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

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("CompiledPlanBlock name must be a non-empty string")
        if not isinstance(self.backend, str) or not self.backend:
            raise TypeError("CompiledPlanBlock backend must be a non-empty string")
        object.__setattr__(self, "spatial", _deep_freeze(self.spatial))
        _evidence(self.spatial, where="CompiledPlanBlock.spatial")


@dataclass(frozen=True, slots=True)
class CompiledPlanRecord:
    """Detached data required by bind/install; never retains Model or Program builders."""

    plan_identity: Identity
    snapshot: Any
    target: str
    backend: str
    layout: Any
    bind_schema: Any
    compile_values: Mapping[Any, Any]
    field_solvers: Mapping[str, Any]
    outputs: tuple[Any, ...]
    diagnostics: tuple[Any, ...]
    requirements: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    blocks: tuple[CompiledPlanBlock, ...]
    time_identity: Any
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
            bind_schema=plan.bind_schema,
            compile_values=plan.compile_values,
            field_solvers=plan.field_solvers,
            outputs=plan.outputs,
            diagnostics=plan.diagnostics,
            requirements=plan.requirements,
            capabilities=plan.capabilities,
            blocks=tuple(
                CompiledPlanBlock(block.name, block.backend, block.spatial)
                for block in plan.blocks
            ),
            time_identity=(
                _evidence(plan.time, where="resolved time")
                if plan.time is not None else None
            ),
        )

    def __post_init__(self) -> None:
        if type(self.plan_identity) is not Identity \
                or self.plan_identity.domain != "resolved-plan":
            raise TypeError("CompiledPlanRecord requires a resolved-plan Identity")
        if self.target not in ("system", "amr_system"):
            raise ValueError("CompiledPlanRecord has an unsupported target")
        object.__setattr__(self, "layout", _deep_freeze(self.layout))
        object.__setattr__(self, "compile_values", _deep_freeze(self.compile_values))
        object.__setattr__(self, "field_solvers", _deep_freeze(self.field_solvers))
        object.__setattr__(self, "outputs", tuple(_deep_freeze(v) for v in self.outputs))
        object.__setattr__(self, "diagnostics", tuple(
            _deep_freeze(v) for v in self.diagnostics))
        object.__setattr__(self, "requirements", _deep_freeze(self.requirements))
        object.__setattr__(self, "capabilities", _deep_freeze(self.capabilities))
        blocks = tuple(self.blocks)
        if not blocks or any(type(block) is not CompiledPlanBlock for block in blocks):
            raise TypeError("CompiledPlanRecord blocks must be exact CompiledPlanBlock values")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(
            self, "contract_identity", make_identity("compiled-plan", self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "resolved_plan_identity": self.plan_identity.to_data(),
            "target": self.target,
            "backend": self.backend,
            "layout": _evidence(self.layout, where="compiled plan layout"),
            "bind_schema": _evidence(self.bind_schema, where="compiled plan bind schema"),
            "compile_values": _evidence(
                self.compile_values, where="compiled plan compile values"),
            "field_solvers": _evidence(
                self.field_solvers, where="compiled plan field solvers"),
            "outputs": _evidence(self.outputs, where="compiled plan outputs"),
            "diagnostics": _evidence(
                self.diagnostics, where="compiled plan diagnostics"),
            "requirements": _evidence(
                self.requirements, where="compiled plan requirements"),
            "capabilities": _evidence(
                self.capabilities, where="compiled plan capabilities"),
            "blocks": [
                {
                    "name": block.name,
                    "backend": block.backend,
                    "spatial": _evidence(
                        block.spatial, where="compiled plan block spatial"),
                }
                for block in self.blocks
            ],
            "time_identity": self.time_identity,
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
    for name in ("target", "backend", "abi_key", "model_hash", "definition_identity"):
        item = getattr(value, name, None)
        if item is not None:
            metadata[name] = _evidence(item, where="%s.%s" % (where, name))
    return {
        "component": _evidence(value, where=where),
        "binary_sha256": digest,
        "metadata": metadata,
    }


def _seal_component(value: Any) -> None:
    seal = getattr(value, "_seal", None)
    if callable(seal) and not getattr(value, "_sealed", False):
        seal()


@dataclass(frozen=True, slots=True)
class CompiledBlockArtifact:
    """One resolved block paired with its authenticated native component."""

    name: str
    model: Any
    spatial: Any

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("CompiledBlockArtifact name must be a non-empty string")
        _binary_evidence(self.model, where="CompiledBlockArtifact.model")
        object.__setattr__(self, "spatial", _deep_freeze(self.spatial))
        _evidence(self.spatial, where="CompiledBlockArtifact.spatial")


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
    artifact_identity: Identity = field(init=False)
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
            if _evidence(compiled.spatial, where="compiled spatial") != _evidence(
                    resolved.spatial, where="resolved spatial"):
                raise ValueError(
                    "compiled block %r changed the resolved spatial descriptor" % compiled.name)
        if self.plan.target == "system" and self.program is None:
            raise ValueError("system CompiledSimulationArtifact requires a compiled program")
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
        evidence = self._current_component_evidence()
        object.__setattr__(self, "_component_evidence", _deep_freeze(evidence))
        object.__setattr__(
            self, "artifact_identity", make_identity("artifact", self._payload(evidence)))

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
    def so_path(self) -> str:
        return str(self._delegate.so_path)

    @property
    def abi_key(self) -> Any:
        return self._delegate.abi_key

    @property
    def cxx(self) -> Any:
        return self._delegate.cxx

    @property
    def std(self) -> Any:
        return self._delegate.std

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
        return getattr(self._delegate, "cache_key", None)

    @property
    def codegen_env(self) -> Any:
        return getattr(self._delegate, "codegen_env", None)

    @property
    def module_manifest(self) -> Any:
        return getattr(self._delegate, "module_manifest", None)

    @property
    def lowering_coverage(self) -> Any:
        return getattr(self._delegate, "lowering_coverage", None)

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

    @property
    def _delegate(self) -> Any:
        return self.program if self.program is not None else self.blocks[0].model

    def _current_component_evidence(self) -> dict[str, Any]:
        evidence = {
            "blocks": [
                {
                    "name": block.name,
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
        }
        return evidence

    def _payload(self, evidence: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan_identity": self.plan.plan_identity.to_data(),
            "target": self.plan.target,
            "components": evidence,
        }

    def verify(self) -> None:
        self.plan.verify()
        current = self._current_component_evidence()
        expected = make_identity("artifact", self._payload(current))
        if self.artifact_identity != expected or self._component_evidence != _deep_freeze(current):
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

    def capability_matrix(self) -> Any:
        return self.manifest().capability_matrix()

    def estimate_memory(self, mesh: Any, *, platform: Any = None, layout: Any = None) -> Any:
        from pops.codegen.inspect_compiled import build_memory_estimate

        return build_memory_estimate(
            self, mesh, platform=platform, layout=layout or self.layout)

    def inspect_amr(self, layout: Any = None) -> Any:
        from pops import inspect_amr

        return inspect_amr(layout or self.layout)

    def scratch_plan(self) -> Any:
        if self.program is None:
            raise ValueError("compiled artifact has no whole-system Program to analyze")
        from pops.codegen.scratch_plan import build_scratch_plan

        return build_scratch_plan(self.program.program)


__all__ = [
    "CompiledBlockArtifact", "CompiledPlanBlock", "CompiledPlanRecord",
    "CompiledSimulationArtifact",
]
