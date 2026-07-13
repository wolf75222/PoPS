"""Generic normalization, assignment and mapping resolution for mesh layouts."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
from typing import Any

from pops.model import Handle, OwnerPath

from ._layout_plan_contracts import (
    LayoutAssignment,
    LayoutHandle,
    LayoutLevel,
    LayoutMappingProvider,
    LayoutMappingRequirement,
    LayoutPlan,
    NormalizedLayout,
    ResolvedLayoutMapping,
    canonical,
    canonical_owner,
    freeze,
    handle_identity,
    json_data,
    name,
    plan_payload,
    subject_kind,
)


def _provider_identity(value: Any, *, where: str) -> tuple[str, Mapping[str, Any]]:
    if isinstance(value, str):
        raise TypeError("%s requires an owner-qualified provider object, never a string" % where)
    projection = getattr(value, "canonical_identity", None)
    if not callable(projection):
        raise TypeError("%s must expose canonical_identity()" % where)
    data = projection()
    if not isinstance(data, Mapping):
        raise TypeError("%s canonical_identity() must return a mapping" % where)
    identity = data.get("qualified_id")
    if not isinstance(identity, str) or not identity:
        raise TypeError("%s canonical identity requires a non-empty qualified_id" % where)
    if getattr(value, "qualified_id", None) != identity:
        raise ValueError("%s canonical identity does not authenticate qualified_id" % where)
    canonical_data = json_data(data, where="%s canonical identity" % where)
    return identity, freeze(canonical_data)


def _descriptor_map(descriptor: Any, method: str) -> dict[str, Any]:
    fn = getattr(descriptor, method, None)
    if not callable(fn):
        raise TypeError("layout descriptor must expose %s()" % method)
    data = json_data(fn(), where="layout descriptor %s()" % method)
    if not isinstance(data, dict):
        raise TypeError("layout descriptor %s() must return a mapping" % method)
    return data


def _descriptor_snapshot(descriptor: Any, *, handle_resolver: Any = None) -> dict[str, Any]:
    """Use the strict Problem projector lazily, avoiding a mesh -> problem import-time cycle."""
    from pops.problem._snapshot import AuthoringSnapshot

    snapshot = AuthoringSnapshot(
        {"descriptor": descriptor}, handle_resolver=handle_resolver).to_dict()["descriptor"]
    if not isinstance(snapshot, dict):
        raise TypeError("layout descriptor snapshot must be a canonical mapping")
    return snapshot


def normalize_layout(handle: LayoutHandle, descriptor: Any, *, handle_resolver: Any = None) \
        -> NormalizedLayout:
    """Project any layout-descriptor implementation onto one common hierarchy representation."""
    if not isinstance(handle, LayoutHandle):
        raise TypeError("normalize_layout requires a canonical LayoutHandle")
    handle_identity(handle, where="normalize_layout layout", kind="layout")
    validate = getattr(descriptor, "validate", None)
    if not callable(validate):
        raise TypeError("layout descriptor must expose validate()")
    validate()
    capabilities = _descriptor_map(descriptor, "capabilities")
    options = _descriptor_map(descriptor, "options")
    requirements = _descriptor_map(descriptor, "requirements")
    snapshot = _descriptor_snapshot(descriptor, handle_resolver=handle_resolver)
    count = capabilities.get("max_levels", capabilities.get("levels", 1))
    ratio = capabilities.get("ratio", 1)
    adaptive = capabilities.get("supports_amr", False)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("layout capabilities require levels/max_levels >= 1")
    if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio < 1:
        raise ValueError("layout capabilities require ratio >= 1")
    if not isinstance(adaptive, bool):
        raise TypeError("layout capability supports_amr must be bool")
    if count == 1:
        ratio = 1
    levels = tuple(LayoutLevel(index, ratio ** index) for index in range(count))
    descriptor_name = getattr(descriptor, "name", type(descriptor).__name__)
    return NormalizedLayout(
        handle=handle,
        descriptor_type="%s.%s" % (type(descriptor).__module__, type(descriptor).__qualname__),
        descriptor_name=str(descriptor_name), adaptive=adaptive, ratio=ratio, levels=levels,
        options=options, capabilities=capabilities, requirements=requirements,
        descriptor_snapshot=snapshot)


class LayoutPlanBuilder:
    """Post-resolution registry that produces one immutable :class:`LayoutPlan`."""

    def __init__(self, owner: Any, *, handle_resolver: Any = None) -> None:
        self._owner = canonical_owner(owner, where="LayoutPlanBuilder.owner")
        self._handle_resolver = handle_resolver
        self._layouts: dict[str, NormalizedLayout] = {}
        self._assignments: dict[tuple[str, str], LayoutAssignment] = {}
        self._requirements: dict[str, LayoutMappingRequirement] = {}

    @property
    def owner(self) -> OwnerPath:
        return self._owner

    def layout(self, local_id: str, descriptor: Any) -> LayoutHandle:
        handle = LayoutHandle(local_id, owner=self._owner)
        if handle.qualified_id in self._layouts:
            raise ValueError("layout %s is already declared" % handle.qualified_id)
        self._layouts[handle.qualified_id] = normalize_layout(
            handle, descriptor, handle_resolver=self._handle_resolver)
        return handle

    def _assign(self, subject: Any, layout: LayoutHandle, kind: str) -> None:
        kind = subject_kind(kind)
        subject_id = handle_identity(subject, where="%s assignment" % kind, kind=kind)
        if not isinstance(layout, LayoutHandle) or layout.qualified_id not in self._layouts:
            raise ValueError("layout assignment requires a LayoutHandle declared by this builder")
        key = (kind, subject_id)
        if key in self._assignments:
            raise ValueError("double layout assignment for %s %s" % key)
        self._assignments[key] = LayoutAssignment(subject, layout)

    def assign_state(self, state: Any, layout: LayoutHandle) -> None:
        self._assign(state, layout, "state")

    def assign_field(self, field: Any, layout: LayoutHandle) -> None:
        self._assign(field, layout, "field")

    def assign_block(self, block: Any, layout: LayoutHandle) -> None:
        self._assign(block, layout, "block")

    def require_mapping(self, source: LayoutHandle, target: LayoutHandle, *, channel: str,
                        reverse: bool = False) -> tuple[LayoutMappingRequirement, ...]:
        for handle in (source, target):
            if not isinstance(handle, LayoutHandle) or handle.qualified_id not in self._layouts:
                raise ValueError("mapping endpoints must be layouts declared by this builder")
        forward = LayoutMappingRequirement(source, target, name(channel, where="mapping channel"))
        rows = [forward]
        if reverse:
            rows.append(LayoutMappingRequirement(target, source, forward.channel,
                                                 forward.qualified_id))
        for row in rows:
            if row.qualified_id in self._requirements:
                raise ValueError("double mapping requirement %s" % row.qualified_id)
            self._requirements[row.qualified_id] = row
        return tuple(rows)

    def resolve(self, *, states: Iterable[Any] = (), fields: Iterable[Any] = (),
                blocks: Iterable[Any] = (),
                providers: Sequence[LayoutMappingProvider] = ()) -> LayoutPlan:
        expected: set[tuple[str, str]] = set()
        for kind, values in (("state", states), ("field", fields), ("block", blocks)):
            for value in values:
                key = (kind, handle_identity(value, where="expected %s" % kind, kind=kind))
                if key in expected:
                    raise ValueError("duplicate expected %s %s" % key)
                expected.add(key)
        authored = set(self._assignments)
        missing, extra = sorted(expected - authored), sorted(authored - expected)
        if missing:
            raise ValueError("unassigned layout subjects: %s" % missing)
        if extra:
            raise ValueError("layout assignments are not exact; unexpected subjects: %s" % extra)

        provider_rows = []
        provider_ids: set[str] = set()
        for provider in providers:
            provider_id, provider_identity = _provider_identity(
                provider, where="layout mapping provider")
            supports = getattr(provider, "supports_layout_mapping", None)
            if not callable(supports):
                raise TypeError("layout mapping provider %s lacks supports_layout_mapping()" %
                                provider_id)
            if provider_id in provider_ids:
                raise ValueError("duplicate mapping provider identity %s" % provider_id)
            provider_ids.add(provider_id)
            provider_rows.append((provider_id, provider_identity, provider))

        resolved = []
        for requirement in sorted(self._requirements.values(), key=lambda row: row.qualified_id):
            matches = []
            for provider_id, provider_identity, provider in provider_rows:
                supported = provider.supports_layout_mapping(requirement)
                if not isinstance(supported, bool):
                    raise TypeError("provider %s supports_layout_mapping() must return bool" %
                                    provider_id)
                if supported:
                    matches.append((provider_id, provider_identity))
            if not matches:
                label = "missing reverse mapping provider" if requirement.reverse_of else \
                    "missing mapping provider"
                raise ValueError("%s for %s -> %s channel %s" % (
                    label, requirement.source.qualified_id, requirement.target.qualified_id,
                    requirement.channel))
            if len(matches) > 1:
                raise ValueError("ambiguous mapping providers for %s: %s" % (
                    requirement.qualified_id,
                    sorted(provider_id for provider_id, _ in matches)))
            provider_id, provider_identity = matches[0]
            resolved.append(ResolvedLayoutMapping(
                requirement, provider_id, provider_identity))

        layouts = tuple(sorted(self._layouts.values(), key=lambda row: row.handle.qualified_id))
        assignments = tuple(sorted(self._assignments.values(),
                                   key=lambda row: (row.subject_kind, row.subject_id)))
        mappings = tuple(sorted(resolved, key=lambda row: row.requirement.qualified_id))
        payload = plan_payload(self._owner, layouts, assignments, mappings)
        canonical_id = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
        return LayoutPlan(self._owner, layouts, assignments, mappings, canonical_id)


def normalize_layout_plan(descriptor: Any, *, owner: Any, local_id: str = "default",
                          states: Iterable[Handle] = (), fields: Iterable[Handle] = (),
                          blocks: Iterable[Handle] = (),
                          handle_resolver: Any = None) -> LayoutPlan:
    """Return the public one-layout degenerate plan, with exact supplied assignments."""
    state_rows, field_rows, block_rows = tuple(states), tuple(fields), tuple(blocks)
    builder = LayoutPlanBuilder(owner, handle_resolver=handle_resolver)
    layout = builder.layout(local_id, descriptor)
    for state in state_rows:
        builder.assign_state(state, layout)
    for field in field_rows:
        builder.assign_field(field, layout)
    for block in block_rows:
        builder.assign_block(block, layout)
    return builder.resolve(states=state_rows, fields=field_rows, blocks=block_rows)


__all__ = [
    "LayoutAssignment", "LayoutHandle", "LayoutLevel", "LayoutMappingProvider",
    "LayoutMappingRequirement", "LayoutPlan", "LayoutPlanBuilder", "NormalizedLayout",
    "ResolvedLayoutMapping", "normalize_layout", "normalize_layout_plan",
]
