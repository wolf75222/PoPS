"""Content-addressed, collision-safe component registration."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ._component_manifest import ComponentManifest
from .component_adapters import ComponentAdapter, adapt_component


@dataclass(frozen=True, slots=True)
class ComponentRecord:
    manifest: ComponentManifest
    adapter: ComponentAdapter = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.manifest != self.adapter.manifest:
            raise ValueError("ComponentRecord manifest and adapter disagree")

    @property
    def component(self) -> Any:
        return self.adapter.component

    @property
    def component_id(self) -> str:
        return self.manifest.component_id

    def to_data(self) -> dict[str, Any]:
        return self.adapter.to_data()


@dataclass(frozen=True, slots=True)
class ComponentRegistrySnapshot:
    revision: int
    records: tuple[ComponentRecord, ...]
    frozen: bool = False
    _by_id: Mapping[str, ComponentRecord] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_by_id", MappingProxyType({record.component_id: record for record in self.records})
        )

    def resolve(self, component_id: Any) -> Any:
        try:
            return self._by_id[component_id].component
        except KeyError:
            raise KeyError("unknown component %r" % (component_id,)) from None

    def manifest(self, component_id: Any) -> ComponentManifest:
        try:
            return self._by_id[component_id].manifest
        except KeyError:
            raise KeyError("unknown component %r" % (component_id,)) from None

    def adapter(self, component_id: Any) -> ComponentAdapter:
        try:
            return self._by_id[component_id].adapter
        except KeyError:
            raise KeyError("unknown component %r" % (component_id,)) from None

    def report(self) -> tuple[dict[str, Any], ...]:
        return tuple(record.to_data() for record in self.records)

    def __len__(self) -> int:
        return len(self.records)


class ComponentRegistry:
    """Atomic registry with deterministic collision and freeze semantics."""

    __slots__ = ("_by_id", "_order", "_revision", "_frozen")

    def __init__(self) -> None:
        self._by_id: dict[str, ComponentRecord] = {}
        self._order: list[str] = []
        self._revision = 0
        self._frozen = False

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def frozen(self) -> bool:
        return self._frozen

    def register(self, component: Any, manifest: ComponentManifest | None = None, *,
                 origin: str = "source", source_uri: str | None = None,
                 entry_points: Mapping[str, Any] | None = None,
                 platform: Mapping[str, Any] | None = None) -> Any:
        """Register a conforming component; an identical repeat is a no-op.

        A manifest may be supplied explicitly.  Otherwise the external component
        exposes ``component_manifest`` as either an attribute or a zero-argument
        method.  All validation completes before registry state is changed.
        """
        if self.frozen:
            raise RuntimeError("ComponentRegistry is frozen; cannot register components")
        if manifest is None:
            manifest = getattr(component, "component_manifest", None)
            if callable(manifest):
                manifest = manifest()
        if not isinstance(manifest, ComponentManifest):
            raise TypeError(
                "component registration requires a ComponentManifest, explicitly or through "
                "component_manifest"
            )
        adapter = adapt_component(
            component, manifest, origin=origin, source_uri=source_uri,
            entry_points=entry_points, platform=platform)
        previous = self._by_id.get(manifest.component_id)
        if previous is not None:
            if previous.manifest.semantic_bytes == manifest.semantic_bytes:
                return previous.component
            raise ValueError(
                "component identity collision for %r: existing semantics %s, incoming semantics %s"
                % (manifest.component_id, previous.manifest.semantic_digest.token,
                   manifest.semantic_digest.token)
            )
        record = ComponentRecord(manifest, adapter)
        self._by_id[manifest.component_id] = record
        self._order.append(manifest.component_id)
        self._revision += 1
        return component

    def resolve(self, component_id: Any) -> Any:
        try:
            return self._by_id[component_id].component
        except KeyError:
            known = ", ".join(self._order) or "<none>"
            raise KeyError(
                "unknown component %r (registered: %s)" % (component_id, known)
            ) from None

    def manifest(self, component_id: Any) -> ComponentManifest:
        try:
            return self._by_id[component_id].manifest
        except KeyError:
            raise KeyError("unknown component %r" % (component_id,)) from None

    def adapter(self, component_id: Any) -> ComponentAdapter:
        try:
            return self._by_id[component_id].adapter
        except KeyError:
            raise KeyError("unknown component %r" % (component_id,)) from None

    def report(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._by_id[item].to_data() for item in self._order)

    def snapshot(self) -> ComponentRegistrySnapshot:
        if not self.frozen:
            raise RuntimeError(
                "ComponentRegistry must be frozen before snapshot capture"
            )
        return ComponentRegistrySnapshot(
            self.revision, tuple(self._by_id[item] for item in self._order), self.frozen
        )

    def freeze(self) -> ComponentRegistry:
        self._frozen = True
        return self

    def __contains__(self, component_id: Any) -> bool:
        return component_id in self._by_id

    def __len__(self) -> int:
        return len(self._order)


__all__ = [
    "ComponentRecord", "ComponentRegistrySnapshot", "ComponentRegistry",
]
