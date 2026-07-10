"""The ordered, name-keyed :class:`OperatorRegistry` (Spec 2).

Insertion order fixes each operator's integer ``OperatorId`` so the C++ codegen
can dispatch by integer in hot kernels while strings stay for debug / validation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any
from weakref import ref

from .hash_data import body_identity, canonical_hash_data
from .handles import Handle, OperatorHandle
from .ownership import MissingOwnershipError, OwnerKind, OwnerPath
from .operators import Operator, validate_operator_signature


def _sha256_fingerprint(namespace: str, payload: Any) -> str:
    canonical = json.dumps(
        canonical_hash_data(payload, where="%s definition fingerprint" % namespace),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return "%s:sha256:%s" % (
        namespace,
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _handle_declaration_fingerprint(handles: Any) -> str:
    """Stable fallback for owner protocols which expose declarations but no Module hash."""
    rows = []
    for handle in handles:
        signature = getattr(handle, "signature", None)
        rows.append({
            "kind": handle.kind,
            "local_id": handle.local_id,
            "schema_version": handle.schema_version,
            "registered_operator_name": getattr(
                handle, "registered_operator_name", None),
            "signature": signature.to_data() if signature is not None else None,
        })
    return _sha256_fingerprint(
        "pops.declarations",
        {"schema": "pops.declarations.v1", "declarations": rows},
    )


def _operator_registry_fingerprint(registry: Any) -> str:
    """Content address of a standalone operator registry, independent of its OwnerPath."""
    operators = []
    for operator in registry:
        operators.append({
            "name": operator.name,
            "kind": operator.kind,
            "signature": operator.signature.to_data(),
            "capabilities": operator.capabilities,
            "requirements": operator.requirements,
            "lowering": operator.lowering,
            "body": body_identity(operator.body),
        })
    return _sha256_fingerprint(
        "pops.operator-registry",
        {
            "schema": "pops.operator-registry.v1",
            "operators": operators,
            "aliases": registry.aliases(),
        },
    )


class DeclarationIndex:
    """Read-only derived index over handles owned by authoritative family registries.

    The index cannot declare anything; it only authenticates that an owner-qualified value names a
    declaration which really exists.  This keeps small per-family registries authoritative while
    giving Case/block qualification one generic protocol.
    """

    __slots__ = ("_owner_path", "_records")

    def __init__(self, *, owner: Any, handles: Any) -> None:
        self._owner_path = OwnerPath.coerce(owner)
        records = {}
        ordered_handles = []
        for handle in handles:
            if not isinstance(handle, Handle):
                raise TypeError("DeclarationIndex handles must be Handle values")
            if handle.owner_path != self._owner_path:
                raise MissingOwnershipError(
                    "declaration %s is owned by %s, not index owner %s"
                    % (handle.qualified_id, handle.owner_path, self._owner_path))
            key = (handle.kind, handle.local_id)
            if key in records:
                raise ValueError(
                    "duplicate %s declaration %r in owner %s"
                    % (handle.kind, handle.local_id, self._owner_path))
            records[key] = handle
            ordered_handles.append(handle)
        if self._owner_path.is_authoring:
            self._owner_path._bind_definition_fingerprint(
                _handle_declaration_fingerprint(ordered_handles),
                priority=10,
            )
        self._records = records

    @property
    def owner_path(self) -> OwnerPath:
        return self._owner_path

    def authenticate(self, handle: Any) -> Handle:
        """Return the canonical registry-issued value or reject owner/membership mismatch."""
        if not isinstance(handle, Handle):
            raise TypeError("declaration authentication requires a Handle")
        expected_owner = (
            self._owner_path.canonical() if handle.is_resolved else self._owner_path
        )
        if handle.owner_path != expected_owner:
            raise MissingOwnershipError(
                "declaration %s is owned by %s, expected %s"
                % (handle.qualified_id, handle.owner_path, expected_owner))
        key = (handle.kind, handle.local_id)
        try:
            registered = self._records[key]
        except KeyError:
            known = ", ".join(
                "%s:%s" % item for item in sorted(self._records)) or "<none>"
            raise MissingOwnershipError(
                "declaration %s is not registered by owner %s (known: %s)"
                % (handle.qualified_id, self._owner_path, known)) from None
        expected = registered._resolved(expected_owner) if handle.is_resolved else registered
        matches = (
            expected.canonical_identity() == handle.canonical_identity()
            if handle.is_resolved
            else expected == handle
        )
        if not matches:
            raise MissingOwnershipError(
                "declaration %s does not match the registry-authenticated identity %s"
                % (handle.qualified_id, registered.qualified_id))
        return registered

    def contains(self, handle: Any) -> bool:
        try:
            self.authenticate(handle)
        except (TypeError, MissingOwnershipError):
            return False
        return True

    def records(self) -> tuple[Handle, ...]:
        return tuple(self._records.values())


class OperatorRegistry:
    """An ordered, name-keyed registry of :class:`Operator` with stable integer ids.

    Insertion order fixes the ``OperatorId`` (``id_of`` / ``by_id``) so the C++
    codegen (S2-6) can dispatch by integer in hot kernels while strings stay for
    debug / validation only. Re-registering an existing name raises.
    """

    def __init__(self, *, owner: Any) -> None:
        self._owner_path = OwnerPath.coerce(owner).require_authoring_root(
            OwnerKind.MODEL_DEFINITION, where="OperatorRegistry owner")
        self._by_name = {}
        self._order = []
        self._aliases = {}
        # A standalone registry must already have a reproducible canonical owner before a Module
        # view exists. Module later replaces this lower-priority provider with its complete hash.
        registry_ref = ref(self)

        def registry_fingerprint() -> str | None:
            registry = registry_ref()
            return None if registry is None else _operator_registry_fingerprint(registry)

        self._owner_path._bind_definition_fingerprint_provider(
            registry_fingerprint, priority=20)

    @property
    def owner_path(self) -> Any:
        """Read-only declaration owner shared by every handle in the registry."""
        return self._owner_path

    def register(self, operator: Any) -> Any:
        """Register ``operator`` and return it; its id is its insertion index."""
        if not isinstance(operator, Operator):
            raise TypeError("register expects an Operator, got %r" % (operator,))
        # Registry is a trust boundary too: Operator remains an internal mutable
        # record for codegen, so revalidate in case a record was modified between
        # construction and registration.
        validate_operator_signature(
            operator.kind, operator.signature, operator_name=operator.name)
        if operator.name in self._aliases:
            raise ValueError(
                "operator %r collides with registered alias targeting %r"
                % (operator.name, self._aliases[operator.name]))
        if operator.name in self._by_name:
            raise ValueError("operator %r already registered" % (operator.name,))
        self._by_name[operator.name] = operator
        self._order.append(operator.name)
        return operator

    def register_alias(self, alias: Any, target: Any) -> str:
        """Declare one immutable public alias for an existing registry operator.

        Alias resolution is registry-authenticated: carrying a different target
        inside an :class:`OperatorHandle` never grants access by itself. Every alias
        declaration is register-once; a repeat, collision, or retarget attempt fails
        loudly. Repeated alias *resolution* through :meth:`target_for_handle` remains
        side-effect free and permitted.
        """
        if not isinstance(alias, str) or not alias:
            raise ValueError("operator alias must be a non-empty string")
        if not isinstance(target, str) or not target:
            raise ValueError("operator alias target must be a non-empty string")
        if target not in self._by_name:
            known = ", ".join(self._order) or "<none>"
            raise ValueError(
                "operator alias %r targets unknown operator %r (registered: %s)"
                % (alias, target, known))
        if alias in self._by_name:
            raise ValueError("operator alias %r collides with a registered operator" % alias)
        existing = self._aliases.get(alias)
        if existing is not None:
            raise ValueError(
                "operator alias %r is already registered for %r; aliases are register-once "
                "and cannot be redeclared for %r"
                % (alias, existing, target))
        self._aliases[alias] = target
        return alias

    def aliases(self) -> Any:
        """Detached ``{public_alias: registered_target}`` declaration table."""
        return dict(self._aliases)

    def declaration_index(self) -> DeclarationIndex:
        """Read-only authentication view of operator and alias handles."""
        handles = []
        for public_name in (*self._order, *self._aliases):
            target = self.target_for_handle(public_name)
            operator = self.get(target)
            handles.append(OperatorHandle(
                public_name,
                kind=operator.kind,
                owner=self.owner_path,
                signature=operator.signature,
                registered_operator_name=target,
            ))
        return DeclarationIndex(owner=self.owner_path, handles=handles)

    def target_for_handle(self, public_name: Any) -> str:
        """Authenticated registry target for a handle's public local identity."""
        if not isinstance(public_name, str) or not public_name:
            raise TypeError("operator handle name must be a non-empty string")
        if public_name in self._by_name:
            return public_name
        try:
            return self._aliases[public_name]
        except KeyError:
            known = self._order + list(self._aliases)
            raise KeyError(
                "unknown operator handle %r (registered operators/aliases: %s)"
                % (public_name, ", ".join(known) or "<none>")) from None

    def get(self, name: Any) -> Any:
        """Return the operator named ``name`` or raise a clear KeyError."""
        try:
            return self._by_name[name]
        except KeyError:
            known = ", ".join(self._order) or "<none>"
            raise KeyError(
                "unknown operator %r (registered: %s)" % (name, known)) from None

    def names(self) -> Any:
        """Operator names in registration (id) order."""
        return list(self._order)

    def operators_of_kind(self, kind: Any) -> Any:
        """Operators of the given kind, in registration order."""
        return [self._by_name[n] for n in self._order if self._by_name[n].kind == kind]

    def default_of_kind(self, kind: Any) -> Any:
        """The default operator of ``kind`` for model-free resolution.

        Picks the operator flagged ``capabilities["default"]`` if there is exactly
        one; otherwise the sole operator of that kind. Raises a clear error when none
        exists, or when several are compatible and none is privileged -- the caller
        must then disambiguate with an explicit ``P.call(name, ...)``.

        This is a BUILD-TIME resolution (kind -> operator), used only while lowering a Program; it is
        never on a hot kernel path. In a generated kernel operators are addressed by their integer
        ``OperatorId`` (:meth:`id_of` / :meth:`by_id`), so no operator-name string lookup survives into
        the compiled step (ADC-528).
        """
        candidates = self.operators_of_kind(kind)
        privileged = [op for op in candidates if op.capabilities.get("default")]
        if len(privileged) == 1:
            return privileged[0]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise KeyError("no %s operator registered" % kind)
        names = ", ".join(op.name for op in candidates)
        raise ValueError(
            "multiple %s operators are compatible (%s); call P.call(name, ...) "
            "explicitly" % (kind, names))

    def id_of(self, name: Any) -> int:
        """Integer OperatorId of ``name`` (its registration index)."""
        return self._order.index(name)

    def by_id(self, operator_id: Any) -> Any:
        """Operator at integer id ``operator_id``."""
        return self._by_name[self._order[operator_id]]

    def __contains__(self, name: Any) -> bool:
        return name in self._by_name or name in self._aliases

    def __iter__(self) -> Any:
        return (self._by_name[n] for n in self._order)

    def __len__(self) -> int:
        return len(self._order)

    def __repr__(self) -> str:
        return "OperatorRegistry(%s)" % ", ".join(self._order)
