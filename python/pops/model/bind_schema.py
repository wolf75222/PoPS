"""Immutable, owner-qualified model parameter binding schema.

Slots use canonical ``ParamHandle`` identities, so two blocks instantiating the
same Module remain distinct. ``hash`` covers the complete binding plan;
``artifact_hash`` excludes runtime defaults and provenance so changing a bind
value does not invalidate the reusable native artifact.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pops.model.handles import ParamHandle
from pops.model.ownership import OwnerKind
from pops._manifest_protocol import (
    manifest_envelope,
    parse_manifest_envelope,
    strict_json_loads,
)

from ._bind_schema_aliases import validate_authoring_aliases, validate_serialized_aliases
from ._bind_slot import BindSlot
from .resolved_bindings import ResolvedBindings


BIND_SCHEMA_VERSION = 2
_MANIFEST_KIND = "bind-schema"
_PAYLOAD_KEYS = {"slots", "aliases"}


def _resolved_bind_data(declaration: Any, resolver: Any) -> dict[str, Any]:
    """Project a declaration after authenticating every expression Handle leaf."""
    bind_data = getattr(declaration, "bind_data", None)
    if not callable(bind_data):
        raise TypeError("parameter declaration must implement canonical bind_data()")
    row = bind_data()
    if row.get("kind") != "derived":
        return row
    expression = getattr(declaration, "expression", None)
    resolve_references = getattr(expression, "resolve_references", None)
    if not callable(resolve_references):
        raise TypeError("DerivedParam expression must expose resolve_references(resolver)")
    resolved = resolve_references(resolver)
    from pops.ir.visitors import _key
    from pops.params._declaration_data import freeze_json, thaw_json

    row["expression"] = {
        "protocol": "pops.expr.key.v1",
        "value": thaw_json(freeze_json(_key(resolved), where="resolved DerivedParam expression")),
    }
    return row


class BindSchema:
    """Canonical immutable table of every qualified parameter slot in a Problem."""

    __slots__ = ("schema_version", "_slots", "_by_handle", "_aliases")

    def __init__(
        self, slots: Any = (), *, aliases: Any = None, _serialized_aliases: Any = None,
    ) -> None:
        try:
            values = tuple(slots)
        except TypeError:
            raise TypeError("BindSchema slots must be an iterable of BindSlot values") from None
        if any(not isinstance(slot, BindSlot) for slot in values):
            raise TypeError("BindSchema slots must contain only BindSlot values")
        expected_ordinals = list(range(len(values)))
        ordinals = [slot.ordinal for slot in values]
        if ordinals != expected_ordinals:
            raise ValueError(
                "BindSchema slot ordinals must be contiguous and ordered %s (got %s)"
                % (expected_ordinals, ordinals)
            )
        qids = [slot.qid for slot in values]
        if len(set(qids)) != len(qids):
            raise ValueError("BindSchema contains duplicate qualified ParamHandle identities")
        by_handle = {slot.handle: slot for slot in values}
        if len(by_handle) != len(values):
            raise ValueError("BindSchema contains duplicate ParamHandle values")
        object.__setattr__(self, "schema_version", BIND_SCHEMA_VERSION)
        object.__setattr__(self, "_slots", values)
        object.__setattr__(self, "_by_handle", MappingProxyType(by_handle))
        if aliases is not None and _serialized_aliases is not None:
            raise TypeError("BindSchema aliases have two competing authorities")
        if _serialized_aliases is not None:
            checked_aliases = validate_serialized_aliases(_serialized_aliases, self._slots)
        else:
            checked_aliases = validate_authoring_aliases(aliases or {}, self._by_handle)
        object.__setattr__(self, "_aliases", checked_aliases)
        self._validate_dependencies()

    @staticmethod
    def _scope_key(handle: ParamHandle) -> Any:
        if handle.is_instance:
            return ("block", handle.block_ref.qualified_id)
        if handle.owner_path.nodes[0].kind is OwnerKind.CASE:
            return ("case", str(handle.owner_path))
        return ("shared", str(handle.owner_path))

    def _validate_dependencies(self) -> None:
        by_scope_name_kind: dict[tuple[Any, str, str], list[BindSlot]] = {}
        for slot in self._slots:
            key = (self._scope_key(slot.handle), slot.handle.local_id, slot.kind)
            by_scope_name_kind.setdefault(key, []).append(slot)
        dependencies_by_slot: dict[BindSlot, tuple[BindSlot, ...]] = {}
        for slot in self._slots:
            resolved_dependencies = []
            for dependency in slot.declaration["depends_on"]:
                key = (
                    self._scope_key(slot.handle),
                    dependency["name"],
                    dependency["param_kind"],
                )
                candidates = by_scope_name_kind.get(key, [])
                if len(candidates) != 1:
                    raise ValueError(
                        "BindSchema cannot resolve dependency %r of %s within the same owner "
                        "scope (matches=%d)" % (dependency["name"], slot.qid, len(candidates))
                    )
                resolved_dependencies.append(candidates[0])
            dependencies_by_slot[slot] = tuple(resolved_dependencies)
            self._validate_derived_phase(slot)
            self._validate_dependency_phases(slot, resolved_dependencies)
            self._validate_expression_dependencies(slot, resolved_dependencies)

        state: dict[BindSlot, int] = {}
        stack: list[BindSlot] = []

        def visit(slot: BindSlot) -> None:
            mark = state.get(slot, 0)
            if mark == 2:
                return
            if mark == 1:
                start = stack.index(slot)
                cycle = stack[start:] + [slot]
                raise ValueError(
                    "BindSchema DerivedParam dependency cycle: %s"
                    % " -> ".join(item.qid for item in cycle)
                )
            state[slot] = 1
            stack.append(slot)
            for dependency in dependencies_by_slot[slot]:
                if dependency.kind == "derived":
                    visit(dependency)
            stack.pop()
            state[slot] = 2

        for slot in self.derived_slots:
            visit(slot)

    @staticmethod
    def _validate_derived_phase(slot: BindSlot) -> None:
        if slot.kind != "derived":
            return
        phase = slot.declaration["phase"]
        storage = slot.declaration["storage"]
        invalidation = slot.declaration["invalidation"]
        if phase == "compile":
            if storage != "inline" or invalidation != "never":
                raise ValueError(
                    "Compile DerivedParam %s requires storage=inline and invalidation=never"
                    % slot.qid
                )
            return
        if phase == "bind":
            if storage != "derived_cache" or invalidation not in ("on_dependencies", "per_bind"):
                raise ValueError(
                    "Bind DerivedParam %s requires storage=derived_cache and "
                    "invalidation=on_dependencies or per_bind" % slot.qid
                )
            return
        raise NotImplementedError(
            "DerivedParam %s phase=%s is declared explicitly but this artifact has no execution "
            "provider for Runtime/PerBlock/PerLevel derived parameters" % (slot.qid, phase)
        )

    @staticmethod
    def _validate_dependency_phases(slot: BindSlot, dependencies: list[BindSlot]) -> None:
        if slot.kind != "derived":
            return
        phase = slot.declaration["phase"]
        for dependency in dependencies:
            if phase == "compile" and dependency.kind == "runtime":
                raise ValueError(
                    "Compile DerivedParam %s cannot depend on runtime parameter %s"
                    % (slot.qid, dependency.qid)
                )
            if dependency.kind != "derived":
                continue
            dependency_phase = dependency.declaration["phase"]
            allowed = {"compile"} if phase == "compile" else {"compile", "bind"}
            if dependency_phase not in allowed:
                raise ValueError(
                    "DerivedParam %s phase=%s cannot depend on %s phase=%s"
                    % (slot.qid, phase, dependency.qid, dependency_phase)
                )

    @staticmethod
    def _validate_expression_dependencies(slot: BindSlot, dependencies: list[BindSlot]) -> None:
        if slot.kind != "derived":
            return
        allowed_local = {dependency.handle.local_id for dependency in dependencies}
        allowed_qid = {dependency.qid for dependency in dependencies}
        for dependency in dependencies:
            if dependency.handle.declaration_ref is not None:
                allowed_qid.add(dependency.handle.declaration_ref.qualified_id)
        for reference_kind, reference in slot.expression_references():
            allowed = allowed_local if reference_kind == "local" else allowed_qid
            if reference not in allowed:
                raise ValueError(
                    "DerivedParam %s expression reads undeclared dependency %r; depends_on must "
                    "authenticate every parameter leaf" % (slot.qid, reference)
                )

    @classmethod
    def from_problem(cls, problem: Any) -> BindSchema:
        """Collect Module parameters per block and CASE-owned control parameters once."""
        block_registry = getattr(problem, "_blocks", None)
        if block_registry is None or not hasattr(block_registry, "items"):
            raise TypeError("BindSchema.from_problem requires a pops.Problem")
        slots: list[BindSlot] = []
        aliases: dict[ParamHandle, ParamHandle] = {}
        for block_name, spec in block_registry.items():
            block = block_registry.handle(block_name)
            model = spec["model"]
            module = model
            if not (
                callable(getattr(module, "params", None))
                and callable(getattr(module, "param_handle", None))
            ):
                module = getattr(model, "module", None)
            if module is None:
                continue
            params = getattr(module, "params", None)
            param_handle = getattr(module, "param_handle", None)
            if not callable(params) or not callable(param_handle):
                raise TypeError(
                    "block %r model parameter surface must be a pops.model.Module" % block_name
                )
            for _, declaration in params().items():
                declaration_handle = param_handle(declaration)
                live_instance = problem.qualify(declaration_handle, block=block)
                canonical = problem.resolve(declaration_handle, block=block)
                if not isinstance(canonical, ParamHandle):
                    raise TypeError("Problem resolved a Module parameter to a non-ParamHandle")
                row = _resolved_bind_data(
                    declaration,
                    lambda handle, selected=block: problem.resolve(handle, block=selected),
                )
                slots.append(BindSlot(len(slots), canonical, row))
                aliases[live_instance] = canonical

        case_registry = getattr(problem, "_param_registry", None)
        handles = getattr(case_registry, "handles", None)
        declaration_for = getattr(case_registry, "declaration", None)
        if callable(handles) and callable(declaration_for):
            for live_handle in handles():
                if not isinstance(live_handle, ParamHandle):
                    raise TypeError("Problem ParamRegistry handles() must return ParamHandle values")
                declaration = declaration_for(live_handle)
                canonical = problem.resolve(live_handle)
                if not isinstance(canonical, ParamHandle):
                    raise TypeError("Problem resolved a Case parameter to a non-ParamHandle")
                slots.append(
                    BindSlot(
                        len(slots), canonical,
                        _resolved_bind_data(declaration, problem.resolve),
                    )
                )
                aliases[live_handle] = canonical
        elif case_registry is not None and len(case_registry):
            raise ValueError(
                "BindSchema cannot consume the transitional ownerless Problem parameter registry; "
                "migrate it to the canonical CASE-owned ParamRegistry"
            )
        return cls(slots, aliases=aliases)

    @property
    def slots(self) -> tuple[BindSlot, ...]:
        return self._slots

    @property
    def runtime_slots(self) -> tuple[BindSlot, ...]:
        return tuple(slot for slot in self._slots if slot.kind == "runtime")

    @property
    def const_slots(self) -> tuple[BindSlot, ...]:
        return tuple(slot for slot in self._slots if slot.kind == "const")

    @property
    def derived_slots(self) -> tuple[BindSlot, ...]:
        return tuple(slot for slot in self._slots if slot.kind == "derived")

    def slot(self, handle: Any) -> BindSlot:
        canonical = self._canonical_handle(handle)
        return self._by_handle[canonical]

    def _canonical_handle(self, handle: Any) -> ParamHandle:
        if not isinstance(handle, ParamHandle):
            raise TypeError(
                "bind parameter keys must be block-qualified ParamHandle values, not %s"
                % type(handle).__name__
            )
        if handle in self._by_handle:
            return handle
        canonical = self._aliases.get(handle.qualified_id)
        if canonical is not None:
            return canonical
        if (
            not handle.is_instance
            and handle.owner_path.nodes[0].kind not in (OwnerKind.SHARED, OwnerKind.CASE)
        ):
            raise ValueError(
                "parameter handle %s is not block-qualified; bind through block[param_handle]"
                % handle.qualified_id
            )
        raise KeyError("parameter handle %s is not present in this BindSchema" % handle.qualified_id)

    def resolve_compile(self) -> Mapping[ParamHandle, Any]:
        """Materialize only Const and Compile-phase DerivedParam slots during resolution."""
        from ._bind_schema_phases import resolve_compile
        return resolve_compile(self)

    def resolve_bind(
        self,
        values: Any,
        *,
        compile_values: Mapping[ParamHandle, Any],
    ) -> Mapping[ParamHandle, Any]:
        """Validate concrete overrides and materialize only Bind-phase derivations."""
        from ._bind_schema_phases import resolve_bind
        return resolve_bind(self, values, compile_values=compile_values)

    def to_dict(self) -> dict[str, Any]:
        return manifest_envelope(
            kind=_MANIFEST_KIND,
            schema_version=self.schema_version,
            payload={
                "slots": [slot.to_dict() for slot in self._slots],
                "aliases": {
                    alias_qid: target.qualified_id
                    for alias_qid, target in sorted(self._aliases.items())
                },
            },
        )

    @classmethod
    def from_dict(cls, data: Any) -> BindSchema:
        row = parse_manifest_envelope(
            data,
            kind=_MANIFEST_KIND,
            schema_version=BIND_SCHEMA_VERSION,
            payload_keys=_PAYLOAD_KEYS,
            where="BindSchema",
        )
        if not isinstance(row["slots"], (list, tuple)):
            raise TypeError("BindSchema slots must be a list")
        result = cls(
            (BindSlot.from_dict(slot) for slot in row["slots"]),
            _serialized_aliases=row["aliases"],
        )
        if result.to_dict() != dict(data):
            raise ValueError("BindSchema payload is not canonical")
        return result

    @classmethod
    def from_json(cls, text: Any) -> BindSchema:
        return cls.from_dict(strict_json_loads(text, where="BindSchema JSON"))

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=None if indent else (",", ":"),
            indent=indent, allow_nan=False,
        )

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def artifact_data(self) -> dict[str, Any]:
        # Authoring aliases contain process-local capability qids. They are authenticated by
        # ``hash`` and restored by the full wire form, but are not compile semantics and must not
        # make an otherwise reusable native artifact depend on one authoring session.
        return manifest_envelope(
            kind=_MANIFEST_KIND,
            schema_version=self.schema_version,
            payload={
                "slots": [slot.artifact_data() for slot in self._slots],
                "aliases": {},
            },
        )

    @property
    def artifact_hash(self) -> str:
        payload = json.dumps(
            self.artifact_data(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self._slots)

    def __iter__(self) -> Any:
        return iter(self._slots)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("BindSchema is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("BindSchema is immutable")

    def __repr__(self) -> str:
        return "BindSchema(slots=%d, runtime=%d, hash=%r)" % (
            len(self), len(self.runtime_slots), self.hash[:12]
        )


__all__ = ["BIND_SCHEMA_VERSION", "BindSchema", "BindSlot", "ResolvedBindings"]
