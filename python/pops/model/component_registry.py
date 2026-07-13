"""Content-addressed, collision-safe component registration."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import inspect
from types import MappingProxyType
from typing import Any

from ._component_manifest import ComponentManifest
from .component_protocols import FACET_PROTOCOLS


@dataclass(frozen=True, slots=True)
class ComponentRecord:
    manifest: ComponentManifest
    component: Any = field(compare=False, repr=False)

    @property
    def component_id(self) -> str:
        return self.manifest.component_id


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

    def register(self, component: Any, manifest: ComponentManifest | None = None) -> Any:
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
        self._validate_facets(component, manifest)
        previous = self._by_id.get(manifest.component_id)
        if previous is not None:
            if previous.manifest.semantic_bytes == manifest.semantic_bytes:
                return previous.component
            raise ValueError(
                "component identity collision for %r: existing semantics %s, incoming semantics %s"
                % (manifest.component_id, previous.manifest.semantic_digest.token,
                   manifest.semantic_digest.token)
            )
        record = ComponentRecord(manifest, component)
        self._by_id[manifest.component_id] = record
        self._order.append(manifest.component_id)
        self._revision += 1
        return component

    @staticmethod
    def _validate_facets(component: Any, manifest: ComponentManifest) -> None:
        unknown = sorted(set(manifest.facets) - set(FACET_PROTOCOLS))
        if unknown:
            raise ValueError("unknown component facet(s): %s" % ", ".join(unknown))
        methods = {
            "requirement": ("requirements", ()),
            "lowering": ("lower", (object(),)),
            "stencil": ("stencil", ()),
            "stability": ("stability", ()),
            "provider": ("providers", ()),
            "effects": ("effects", ()),
            "restart": ("restart", ()),
            "report": ("report", ()),
            "fallible_evaluation": ("evaluate", (object(),)),
        }
        malformed = []
        for facet in manifest.facets:
            if not isinstance(component, FACET_PROTOCOLS[facet]):
                malformed.append(facet)
                continue
            method_name, probe_args = methods[facet]
            method = getattr(component, method_name, None)
            if not callable(method):
                malformed.append(facet)
                continue
            try:
                inspect.signature(method).bind(*probe_args)
            except (TypeError, ValueError):
                malformed.append(facet)
        if malformed:
            raise TypeError(
                "component %r does not conform to advertised facet(s): %s"
                % (manifest.component_id, ", ".join(malformed))
            )

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
