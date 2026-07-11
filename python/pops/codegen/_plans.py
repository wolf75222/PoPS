"""Immutable compiler and install plans for the public ``pops.compile`` route.

``ResolvedPlan`` is short-lived compiler input.  It may point at deeply frozen model/program IR,
but never at the ``Problem`` facade or any registry.  ``InstallPlan`` is the smaller value retained
by the compiled artifact: block-native loaders plus detached runtime descriptors.  Bind consumes it
directly and has no fallback that can reconstruct state from authoring builders.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def _mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("%s must be a mapping" % where)
    result = dict(value)
    if any(not isinstance(key, str) or not key for key in result):
        raise TypeError("%s keys must be non-empty strings" % where)
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class ResolvedBlock:
    """One compiler-owned block selection before its native loader is built."""

    name: str
    model: Any
    spatial: Any

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("ResolvedBlock name must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ResolvedPlan:
    """The sole typed input accepted by public orchestration compilers."""

    snapshot: Any
    target: str
    layout: Any
    time: Any
    blocks: tuple[ResolvedBlock, ...]
    bind_schema: Any
    field_solvers: Mapping[str, Any]
    outputs: tuple[Any, ...]
    diagnostics: tuple[Any, ...]
    libraries: tuple[Any, ...]

    def __post_init__(self) -> None:
        if self.target not in ("system", "amr_system"):
            raise ValueError("ResolvedPlan target must be 'system' or 'amr_system'")
        if not self.blocks:
            raise ValueError("ResolvedPlan requires at least one block")
        names = [block.name for block in self.blocks]
        if len(set(names)) != len(names):
            raise ValueError("ResolvedPlan block names must be unique")
        object.__setattr__(
            self, "field_solvers", _mapping(self.field_solvers, where="ResolvedPlan field_solvers"))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "libraries", tuple(self.libraries))

    @property
    def first_model(self) -> Any:
        return self.blocks[0].model


@dataclass(frozen=True, slots=True)
class InstallBlock:
    """One block-native loader and its detached spatial selection."""

    name: str
    model: Any
    spatial: Any

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("InstallBlock name must be a non-empty string")
        if self.model is None:
            raise ValueError("InstallBlock %r has no compiled model" % self.name)


@dataclass(frozen=True, slots=True)
class InstallPlan:
    """Deeply immutable runtime plan retained by a public compiled artifact."""

    snapshot_hash: str
    target: str
    layout: Any
    blocks: tuple[InstallBlock, ...]
    bind_schema: Any
    field_solvers: Mapping[str, Any]
    outputs: tuple[Any, ...]
    diagnostics: tuple[Any, ...]
    has_program: bool

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_hash, str) or len(self.snapshot_hash) != 64:
            raise ValueError("InstallPlan snapshot_hash must be a sha256 hex digest")
        if self.target not in ("system", "amr_system"):
            raise ValueError("InstallPlan target must be 'system' or 'amr_system'")
        blocks = tuple(self.blocks)
        if not blocks:
            raise ValueError("InstallPlan requires at least one block")
        names = [block.name for block in blocks]
        if len(set(names)) != len(names):
            raise ValueError("InstallPlan block names must be unique")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(
            self, "field_solvers", _mapping(self.field_solvers, where="InstallPlan field_solvers"))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "has_program", bool(self.has_program))

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def block_models(self) -> Mapping[str, Any]:
        return MappingProxyType({block.name: block.model for block in self.blocks})

    def assemble_instances(self, initial: Any) -> dict[str, dict[str, Any]]:
        """Materialize one fresh runtime input mapping from immutable block records."""
        if not isinstance(initial, Mapping):
            raise TypeError("pops.bind: initial_state must be a block-name mapping")
        declared = {block.name for block in self.blocks}
        unknown = sorted(set(initial) - declared)
        if unknown:
            raise ValueError(
                "pops.bind: initial state for unknown block(s) %s; declared blocks: %s"
                % (unknown, sorted(declared))
            )
        result = {}
        for block in self.blocks:
            entry = {"model": block.model, "spatial": block.spatial}
            if block.name in initial:
                entry["initial"] = initial[block.name]
            result[block.name] = entry
        return result


def require_install_plan(compiled: Any) -> InstallPlan:
    """Return the artifact plan or fail; there is intentionally no live-authoring fallback."""
    plan = getattr(compiled, "install_plan", None)
    if not isinstance(plan, InstallPlan):
        raise TypeError(
            "pops.bind: compiled artifact has no immutable InstallPlan; build it with "
            "pops.compile(problem, layout=...)"
        )
    snapshot = getattr(compiled, "authoring_snapshot", None)
    if snapshot is None or getattr(snapshot, "hash", None) != plan.snapshot_hash:
        raise ValueError(
            "pops.bind: InstallPlan does not authenticate against the artifact's "
            "AuthoringSnapshot"
        )
    return plan


__all__ = [
    "InstallBlock", "InstallPlan", "ResolvedBlock", "ResolvedPlan", "require_install_plan",
]
