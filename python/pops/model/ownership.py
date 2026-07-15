"""Typed ownership identities for semantic declarations.

Authoring and canonical ownership are deliberately two different phases of the same immutable
value.  An authoring path carries an opaque process-local authority capability so two builders with
the same display name cannot authenticate each other's declarations.  Resolution returns a new
canonical path with that capability removed.  Only canonical paths are serialisable.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import count
import re
from typing import Any
from urllib.parse import quote


class OwnershipError(ValueError):
    """Base class for invalid declaration ownership or qualification."""


class MissingOwnershipError(OwnershipError):
    """A declaration has no owner registered in the requested scope."""


class DoubleOwnershipError(OwnershipError):
    """A declaration already belongs to a different instance scope."""


class AmbiguousReferenceError(OwnershipError):
    """An unqualified declaration has more than one valid instance owner."""


class IdentityCollisionError(OwnershipError):
    """Distinct authoring authorities claim the same canonical owner identity."""


class UnresolvedOwnershipError(OwnershipError):
    """A process-local authoring owner was used where canonical identity is required."""


class OwnerKind(Enum):
    """Semantic node kinds understood by the core ownership topology."""

    CASE = "case"
    MODEL_DEFINITION = "model_definition"
    BLOCK = "block"
    LAYOUT = "layout"
    DESCRIPTOR = "descriptor"
    CONSUMER = "consumer"
    SHARED = "shared"


@dataclass(frozen=True, slots=True)
class OwnerSegment:
    """One typed node of an ownership path."""

    kind: OwnerKind
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OwnerKind):
            raise TypeError("OwnerSegment.kind must be an OwnerKind")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("OwnerSegment.name must be a non-empty string")

    def to_data(self) -> dict[str, str]:
        return {"kind": self.kind.value, "name": self.name}


_AUTHORITY_SEQUENCE = count()
_FINGERPRINT_RE = re.compile(r"^[a-z][a-z0-9_.-]*:sha256:[0-9a-f]{64}$")


def _validate_definition_fingerprint(value: Any) -> str:
    if not isinstance(value, str) or _FINGERPRINT_RE.fullmatch(value) is None:
        raise ValueError(
            "model-definition fingerprint must have the form "
            "'<namespace>:sha256:<64 lowercase hex digits>'"
        )
    return value


class _AuthoringAuthority:
    """Opaque equality/hash capability issued only by :meth:`OwnerPath.fresh`."""

    __slots__ = (
        "serial", "_fingerprint", "_fingerprint_priority", "_fingerprint_providers",
        "_resolving",
    )

    def __init__(self) -> None:
        self.serial = next(_AUTHORITY_SEQUENCE)
        self._fingerprint = None
        self._fingerprint_priority = -1
        self._fingerprint_providers = []
        self._resolving = False

    def bind_fingerprint(self, fingerprint: Any, *, priority: int) -> None:
        """Install a deterministic fallback when no richer definition provider is bound."""
        value = _validate_definition_fingerprint(fingerprint)
        if priority >= self._fingerprint_priority:
            self._fingerprint = value
            self._fingerprint_priority = priority

    def bind_provider(self, provider: Any, *, priority: int) -> None:
        """Bind the richest available live content fingerprint provider."""
        if not callable(provider):
            raise TypeError("definition fingerprint provider must be callable")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise TypeError("definition fingerprint provider priority must be an integer")
        self._fingerprint_providers.append((priority, provider))

    def fingerprint(self) -> str | None:
        if self._resolving:
            raise UnresolvedOwnershipError(
                "model-definition fingerprint recursively depends on its own OwnerPath"
            )
        self._resolving = True
        try:
            # Providers are weakly backed by their authoritative registries/builders. Retaining
            # older providers lets an unpublished transactional candidate disappear without
            # poisoning the shared owner authority it briefly used.
            for _, provider in sorted(
                    self._fingerprint_providers, key=lambda item: item[0], reverse=True):
                value = provider()
                if value is not None:
                    return _validate_definition_fingerprint(value)
            return self._fingerprint
        finally:
            self._resolving = False


_ROOT_KINDS = frozenset({
    OwnerKind.CASE,
    OwnerKind.MODEL_DEFINITION,
    OwnerKind.LAYOUT,
    OwnerKind.DESCRIPTOR,
    OwnerKind.CONSUMER,
    OwnerKind.SHARED,
})

_TRANSITIONS = {
    OwnerKind.CASE: frozenset({
        OwnerKind.BLOCK, OwnerKind.LAYOUT, OwnerKind.DESCRIPTOR, OwnerKind.CONSUMER,
    }),
    OwnerKind.BLOCK: frozenset({OwnerKind.MODEL_DEFINITION, OwnerKind.DESCRIPTOR}),
    OwnerKind.MODEL_DEFINITION: frozenset({OwnerKind.DESCRIPTOR, OwnerKind.CONSUMER}),
    OwnerKind.LAYOUT: frozenset({OwnerKind.DESCRIPTOR, OwnerKind.CONSUMER}),
    OwnerKind.DESCRIPTOR: frozenset({OwnerKind.DESCRIPTOR, OwnerKind.CONSUMER}),
    OwnerKind.CONSUMER: frozenset({OwnerKind.DESCRIPTOR}),
    OwnerKind.SHARED: frozenset({OwnerKind.DESCRIPTOR, OwnerKind.CONSUMER}),
}


@dataclass(frozen=True, slots=True, init=False)
class OwnerPath:
    """Immutable typed owner hierarchy with explicit authoring/canonical phases.

    Equality and hashing include the opaque authoring authority when present.  Consequently two
    same-named builders remain distinct dictionary keys.  :meth:`canonical` returns a new,
    deterministic path used by resolved handles, manifests and snapshots; it never mutates a live
    authoring key.
    """

    nodes: tuple[OwnerSegment, ...]
    _authority: _AuthoringAuthority | None
    _definition_fingerprint: str | None

    def __init__(
        self,
        *nodes: Any,
        _authority: _AuthoringAuthority | None = None,
        _definition_fingerprint: str | None = None,
    ) -> None:
        if len(nodes) == 1 and isinstance(nodes[0], (tuple, list)):
            nodes = tuple(nodes[0])
        if not nodes or any(not isinstance(node, OwnerSegment) for node in nodes):
            raise TypeError("OwnerPath requires one or more OwnerSegment values")
        if _authority is not None and not isinstance(_authority, _AuthoringAuthority):
            raise TypeError("OwnerPath authoring authority is internal and cannot be supplied")
        normalized = tuple(nodes)
        self._validate_topology(normalized)
        if _definition_fingerprint is not None:
            _validate_definition_fingerprint(_definition_fingerprint)
            if not any(node.kind is OwnerKind.MODEL_DEFINITION for node in normalized):
                raise OwnershipError(
                    "a definition fingerprint requires a model_definition segment"
                )
        object.__setattr__(self, "nodes", normalized)
        object.__setattr__(self, "_authority", _authority)
        object.__setattr__(self, "_definition_fingerprint", _definition_fingerprint)

    @staticmethod
    def _validate_topology(nodes: tuple[OwnerSegment, ...]) -> None:
        if nodes[0].kind not in _ROOT_KINDS:
            raise OwnershipError(
                "OwnerPath cannot start with %s" % nodes[0].kind.value)
        for parent, child in zip(nodes, nodes[1:], strict=False):
            if child.kind not in _TRANSITIONS[parent.kind]:
                raise OwnershipError(
                    "invalid OwnerPath transition %s -> %s"
                    % (parent.kind.value, child.kind.value))

    @classmethod
    def coerce(cls, owner: Any) -> OwnerPath:
        if isinstance(owner, cls):
            return owner
        path = getattr(owner, "owner_path", None)
        if isinstance(path, cls):
            return path
        raise TypeError(
            "owner must be an OwnerPath or expose an OwnerPath owner_path (got %r)" % (owner,))

    @classmethod
    def root(cls, kind: Any, name: Any) -> OwnerPath:
        return cls(OwnerSegment(kind, name))

    @classmethod
    def fresh(cls, kind: Any, name: Any) -> OwnerPath:
        """Issue a process-local authoring authority for one new semantic owner."""
        return cls(OwnerSegment(kind, name), _authority=_AuthoringAuthority())

    @classmethod
    def case(cls, name: Any) -> OwnerPath:
        return cls.root(OwnerKind.CASE, name)

    @classmethod
    def model(cls, name: Any) -> OwnerPath:
        return cls.root(OwnerKind.MODEL_DEFINITION, name)

    @classmethod
    def layout(cls, name: Any) -> OwnerPath:
        return cls.root(OwnerKind.LAYOUT, name)

    @classmethod
    def descriptor(cls, name: Any) -> OwnerPath:
        return cls.root(OwnerKind.DESCRIPTOR, name)

    @classmethod
    def consumer(cls, name: Any) -> OwnerPath:
        return cls.root(OwnerKind.CONSUMER, name)

    @classmethod
    def shared(cls, name: Any) -> OwnerPath:
        return cls.root(OwnerKind.SHARED, name)

    @property
    def is_authoring(self) -> bool:
        return self._authority is not None

    @property
    def is_canonical(self) -> bool:
        return self._authority is None

    @property
    def kind(self) -> OwnerKind:
        return self.nodes[-1].kind

    @property
    def name(self) -> str:
        return self.nodes[-1].name

    @property
    def segments(self) -> tuple[str, ...]:
        """Flat read-only inspection view; equality never relies on this projection."""
        return tuple(value for node in self.nodes for value in (node.kind.value, node.name))

    def child(self, kind: Any, name: Any) -> OwnerPath:
        return OwnerPath(
            self.nodes + (OwnerSegment(kind, name),),
            _authority=self._authority,
            _definition_fingerprint=self._definition_fingerprint,
        )

    @property
    def definition_fingerprint(self) -> str | None:
        """Stable content identity of the contained model definition, when present."""
        if self._definition_fingerprint is not None:
            return self._definition_fingerprint
        if self._authority is not None and self.contains(OwnerKind.MODEL_DEFINITION):
            return self._authority.fingerprint()
        return None

    def _bind_definition_fingerprint(self, fingerprint: Any, *, priority: int = 10) -> None:
        """Internal authoring hook used by authoritative definition registries."""
        self.require_authoring_root(
            OwnerKind.MODEL_DEFINITION, where="definition fingerprint owner"
        )
        authority = self._authority
        if authority is None:
            raise RuntimeError("authoring OwnerPath lost its authority")
        authority.bind_fingerprint(fingerprint, priority=priority)

    def _bind_definition_fingerprint_provider(
        self,
        provider: Any,
        *,
        priority: int = 10,
    ) -> None:
        """Internal hook for a live deterministic definition-hash provider."""
        self.require_authoring_root(
            OwnerKind.MODEL_DEFINITION, where="definition fingerprint owner"
        )
        authority = self._authority
        if authority is None:
            raise RuntimeError("authoring OwnerPath lost its authority")
        authority.bind_provider(provider, priority=priority)

    def contains(self, kind: Any) -> bool:
        if not isinstance(kind, OwnerKind):
            raise TypeError("OwnerPath.contains expects an OwnerKind")
        return any(node.kind is kind for node in self.nodes)

    def require_authoring_root(
        self,
        kind: Any,
        *,
        name: Any = None,
        where: str = "owner",
    ) -> OwnerPath:
        """Authenticate one live mutable declaration authority.

        Canonical paths are deliberately easy to reconstruct from a manifest or snapshot.  They
        identify resolved data, but therefore cannot authorize new declarations.  Mutable builders
        and registries use this guard to require the opaque capability issued by :meth:`fresh`, an
        exact one-node root topology, and (when the builder has its own name) the same semantic name.
        """
        if not isinstance(kind, OwnerKind):
            raise TypeError("require_authoring_root kind must be an OwnerKind")
        if not isinstance(where, str) or not where:
            raise TypeError("require_authoring_root where must be a non-empty string")
        if not self.is_authoring:
            raise UnresolvedOwnershipError(
                "%s must be a live authoring OwnerPath; canonical owner %s identifies resolved "
                "data and cannot authorize declarations" % (where, self))
        if len(self.nodes) != 1 or self.kind is not kind:
            raise OwnershipError(
                "%s must be a root %s authoring OwnerPath, got %s"
                % (where, kind.value, self))
        if name is not None:
            if not isinstance(name, str) or not name:
                raise TypeError("require_authoring_root name must be a non-empty string")
            if self.name != name:
                raise OwnershipError(
                    "%s name %r does not match declaration name %r"
                    % (where, self.name, name))
        return self

    def canonical(self) -> OwnerPath:
        """Return the deterministic, content-addressed resolved owner topology.

        A live model definition must first be attached to an authoritative structural hash
        provider.  Falling back to its display name (or to the process-local authoring serial) would
        make homonymous foreign definitions authenticate as each other or break reproducibility.
        """
        if not self.is_authoring:
            return self
        fingerprint = self.definition_fingerprint
        if self.contains(OwnerKind.MODEL_DEFINITION) and fingerprint is None:
            raise UnresolvedOwnershipError(
                "authoring model owner %s has no deterministic definition fingerprint; register "
                "its declarations through Module, OperatorRegistry, or DeclarationIndex before "
                "canonical resolution" % self
            )
        return OwnerPath(
            self.nodes,
            _definition_fingerprint=fingerprint,
        )

    def presentation(self) -> OwnerPath:
        """Authority-free inspection path which never claims registry authentication."""
        return OwnerPath(
            self.nodes,
            _definition_fingerprint=self.definition_fingerprint,
        )

    def instance_of(self, declaration_owner: Any) -> OwnerPath:
        """Attach one model definition below a block while retaining case authority."""
        if self.kind is not OwnerKind.BLOCK:
            raise OwnershipError("only a block OwnerPath can instantiate a declaration owner")
        declaration = OwnerPath.coerce(declaration_owner)
        if declaration.contains(OwnerKind.BLOCK):
            raise DoubleOwnershipError(
                "cannot instantiate owner %s: it is already block-qualified" % declaration)
        canonical_declaration = declaration.canonical()
        if canonical_declaration.nodes[0].kind not in (
            OwnerKind.MODEL_DEFINITION,
            OwnerKind.DESCRIPTOR,
        ):
            raise OwnershipError(
                "block instances require a model-definition or descriptor owner; got %s"
                % declaration)
        return OwnerPath(
            self.nodes + canonical_declaration.nodes,
            _authority=self._authority,
            _definition_fingerprint=canonical_declaration.definition_fingerprint,
        )

    def to_data(self) -> dict[str, Any]:
        """Canonical JSON-ready representation; authoring capabilities never leak."""
        if self.is_authoring:
            raise UnresolvedOwnershipError(
                "authoring OwnerPath %s must be resolved with canonical() before serialization"
                % self)
        return {
            "schema_version": 2,
            "nodes": [node.to_data() for node in self.nodes],
            "definition_fingerprint": self._definition_fingerprint,
        }

    @classmethod
    def from_data(cls, data: Any) -> OwnerPath:
        """Strict inverse of :meth:`to_data`; no string or legacy shape is accepted."""
        required = {"schema_version", "nodes", "definition_fingerprint"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError(
                "OwnerPath data must contain exactly schema_version, nodes, and "
                "definition_fingerprint"
            )
        schema_version = data["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise TypeError("OwnerPath schema_version must be an integer")
        if schema_version != 2:
            raise ValueError("unsupported OwnerPath schema_version %r" % schema_version)
        fingerprint = data["definition_fingerprint"]
        if fingerprint is not None:
            _validate_definition_fingerprint(fingerprint)
        raw_nodes = data["nodes"]
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise TypeError("OwnerPath nodes must be a non-empty list")
        nodes = []
        for index, raw in enumerate(raw_nodes):
            if not isinstance(raw, Mapping) or set(raw) != {"kind", "name"}:
                raise TypeError(
                    "OwnerPath node %d must contain exactly kind and name" % index)
            if not isinstance(raw["kind"], str):
                raise TypeError("OwnerPath node %d kind must be a string" % index)
            try:
                kind = OwnerKind(raw["kind"])
            except ValueError as exc:
                raise ValueError(
                    "OwnerPath node %d has unknown kind %r" % (index, raw["kind"])) from exc
            nodes.append(OwnerSegment(kind, raw["name"]))
        return cls(nodes, _definition_fingerprint=fingerprint)

    def __str__(self) -> str:
        segments = []
        for node in self.nodes:
            segment = "%s:%s" % (node.kind.value, quote(node.name, safe=""))
            # Authoring paths are capability identities. Do not execute a mutable definition
            # provider just to display one; only canonical paths render their content fingerprint.
            if node.kind is OwnerKind.MODEL_DEFINITION \
                    and self._definition_fingerprint is not None:
                segment += "@%s" % quote(self._definition_fingerprint, safe="")
            segments.append(segment)
        path = "/".join(segments)
        if self._authority is not None:
            path += "#authoring=%d" % self._authority.serial
        return path


__all__ = [
    "AmbiguousReferenceError",
    "DoubleOwnershipError",
    "IdentityCollisionError",
    "MissingOwnershipError",
    "OwnerKind",
    "OwnerPath",
    "OwnerSegment",
    "OwnershipError",
    "UnresolvedOwnershipError",
]
