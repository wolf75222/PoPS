"""Concrete immutable compiled-simulation artifact returned by the public compile phase."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

from pops.identity import Identity, make_identity

from ._plans import ResolvedSimulationPlan, _deep_freeze, _evidence


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

    plan: ResolvedSimulationPlan
    program: Any | None
    blocks: tuple[CompiledBlockArtifact, ...]
    artifact_identity: Identity = field(init=False)
    _component_evidence: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.plan) is not ResolvedSimulationPlan:
            raise TypeError(
                "CompiledSimulationArtifact.plan must be an exact ResolvedSimulationPlan")
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
    def program_param_routes(self) -> Any:
        if self.program is None:
            return None
        return getattr(self.program, "program_param_routes", None)

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
        return self._delegate.inspect()

    def requirements(self) -> Any:
        return self._delegate.requirements()

    def manifest(self) -> Any:
        return self._delegate.manifest()

    def arguments(self) -> Any:
        return self._delegate.arguments()

    def capability_matrix(self) -> Any:
        return self._delegate.capability_matrix()


__all__ = ["CompiledBlockArtifact", "CompiledSimulationArtifact"]
