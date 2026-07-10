"""Typed operator handles (Spec 5 sec.14.2.3).

An :class:`OperatorHandle` is a lightweight, typed reference to a declared operator: it carries
the operator ``name``, its ``kind`` (the operator-first kind, when the declarer supplies it), the
declared :class:`pops.model.signatures.Signature` and the mathematical ``category`` (ADC-559) -- and
nothing else: no numerics, no IR, no Program array data. A user-facing operator declarer
(``m.rate`` / ``m.field_solve`` / ``m.local_linear_map`` / ``m.source_term`` / ``m.linear_source``)
returns one so a named operator is referenced as a typed, INSPECTABLE object, NOT a bare string::

    R = m.rate("explicit_rhs", flux=True, sources=["electric"])
    R.category        # "rate"
    R.signature       # Signature((U, Fields) -> Rate(U))
    rate = P.call(R, U, fields)      # the handle is the one public P.call selector
    rate = R(U, fields)              # ADC-560: the callable facade -> the same IR

The handle is the public :meth:`pops.time.Program.call` selector: the public ``P.call`` REQUIRES an
``OperatorHandle`` (a bare string operator name is refused). ``OperatorHandle.__call__`` (ADC-560) is
a thin FACADE over that same path: it locates the Program from its ProgramValue arguments and delegates
to ``P.call(self, ...)``, so ``R(U, f)`` builds the BYTE-IDENTICAL IR (same ``_ir_hash``) as
``P.call(R, U, f)`` -- same exact owner/kind/signature checks, same registry lookup, ZERO numerics.
The handle holds no Program reference, but it works only in a Program bound to its exact declaring
owner; a homonymous registry from another model is rejected.

This module imports only the standard library (and the sibling ``pops.model`` types lazily inside
methods) so it stays codegen-free and ``_pops``-free and keeps the ``pops.time`` import graph acyclic
(``pops.time`` already imports ``pops.model``; ``pops.model`` never imports ``pops.time``).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any
from urllib.parse import quote


_OWNER_SEQUENCE = count()


@dataclass(frozen=True, slots=True, init=False)
class OwnerPath:
    """Structured, immutable owner identity for a declared object.

    Segments are never flattened for equality/hashing, so ``("a", "b.c")``
    cannot collide with ``("a.b", "c")``.  ADC-653 extends qualification
    across Case/block registries; this value type is the invariant those
    registries share.
    """

    segments: tuple[str, ...]
    _authoring_token: int | None

    def __init__(self, *segments: Any, _authoring_token: int | None = None) -> None:
        if len(segments) == 1 and isinstance(segments[0], (tuple, list)):
            segments = tuple(segments[0])
        if not segments or any(not isinstance(segment, str) or not segment for segment in segments):
            raise ValueError("OwnerPath requires one or more non-empty string segments")
        normalized = tuple(segments)
        if (_authoring_token is not None
                and (isinstance(_authoring_token, bool)
                     or not isinstance(_authoring_token, int)
                     or _authoring_token < 0)):
            raise ValueError("OwnerPath authoring token must be a non-negative integer")
        object.__setattr__(self, "segments", normalized)
        object.__setattr__(self, "_authoring_token", _authoring_token)

    @classmethod
    def coerce(cls, owner: Any) -> "OwnerPath":
        if isinstance(owner, cls):
            return owner
        path = getattr(owner, "owner_path", None)
        if isinstance(path, cls):
            return path
        raise TypeError(
            "owner must be an OwnerPath or expose an OwnerPath owner_path (got %r)" % (owner,))

    @classmethod
    def model(cls, name: Any) -> "OwnerPath":
        return cls("model", name)

    @classmethod
    def problem(cls, name: Any) -> "OwnerPath":
        return cls("problem", name)

    @classmethod
    def program(cls, name: Any) -> "OwnerPath":
        return cls("program", name)

    @classmethod
    def fresh(cls, kind: Any, name: Any) -> "OwnerPath":
        """Create a distinct authoring owner before ADC-653 snapshot qualification.

        The token is process-local authoring identity, never a manifest digest.  ADC-653
        replaces it with registry/snapshot qualification before canonical encoding.
        """
        return cls(kind, name, _authoring_token=next(_OWNER_SEQUENCE))

    def child(self, *segments: Any) -> "OwnerPath":
        if not segments or any(not isinstance(segment, str) or not segment for segment in segments):
            raise ValueError("OwnerPath.child requires non-empty string segments")
        return OwnerPath(
            self.segments + tuple(segments),
            _authoring_token=self._authoring_token,
        )

    def canonical_declaration_path(self) -> "OwnerPath":
        """Drop the process-local authoring token for manifest/snapshot identity."""
        return OwnerPath(self.segments) if self._authoring_token is not None else self

    def __str__(self) -> str:
        path = "/".join(quote(segment, safe="") for segment in self.segments)
        # ``#`` is always percent-encoded inside a user segment, so this structural suffix cannot
        # collide with a legitimate path. It is intentionally absent from canonical manifests.
        if self._authoring_token is not None:
            path += "#authoring=%d" % self._authoring_token
        return path


class Handle:
    """Immutable, owner-qualified identity of one declared object."""

    __slots__ = ("owner_path", "local_id", "kind", "schema_version")
    __pops_ir_immutable__ = True

    def __init__(
        self,
        local_id: Any,
        *,
        kind: Any,
        owner: Any,
        schema_version: int = 1,
    ) -> None:
        if not isinstance(local_id, str) or not local_id:
            raise ValueError("Handle local_id must be a non-empty string")
        if not isinstance(kind, str) or not kind:
            raise ValueError("Handle kind must be a non-empty string")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int) \
                or schema_version < 1:
            raise ValueError("Handle schema_version must be an integer >= 1")
        object.__setattr__(self, "owner_path", OwnerPath.coerce(owner))
        object.__setattr__(self, "local_id", local_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def name(self) -> str:
        return self.local_id

    @property
    def expression_readable(self) -> bool:
        """Whether an explicit ValueExpr may read this declaration as a symbolic value."""
        return True

    @property
    def qualified_id(self) -> str:
        return self._qualified_id(self.owner_path)

    def _qualified_id(self, owner_path: OwnerPath) -> str:
        return "pops.handle.v%d::%s::%s::%s" % (
            self.schema_version,
            owner_path,
            quote(self.kind, safe=""),
            quote(self.local_id, safe=""),
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "local_id": self.local_id,
            "owner_path": self.owner_path.segments,
            "qualified_id": self.qualified_id,
            "schema_version": self.schema_version,
        }

    def canonical_identity(self) -> dict[str, Any]:
        """JSON-ready declaration identity with ephemeral authoring tokens removed."""
        owner = self.owner_path.canonical_declaration_path()
        return {
            "kind": self.kind,
            "local_id": self.local_id,
            "owner_path": owner.segments,
            "qualified_id": self._qualified_id(owner),
            "schema_version": self.schema_version,
        }

    def _identity(self) -> tuple[Any, ...]:
        return (self.schema_version, self.owner_path, self.kind, self.local_id)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Handle) and self._identity() == other._identity()

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self._identity())

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("%s is immutable" % type(self).__name__)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("%s is immutable" % type(self).__name__)

    def __repr__(self) -> str:
        return "%s(local_id=%r, kind=%r, owner=%r)" % (
            type(self).__name__, self.local_id, self.kind, self.owner_path.segments)


class OperatorHandle(Handle):
    """A typed, INSPECTABLE reference to a declared operator (Spec 5 sec.14.2.3, ADC-559).

    Carries the operator ``name``, an optional ``kind`` (the operator-first kind, e.g.
    ``"local_rate"`` / ``"local_source"`` / ``"local_linear_operator"`` / ``"field_operator"``),
    the declared :class:`pops.model.signatures.Signature` (or ``None`` when the declarer did not
    supply one) and the mathematical ``category`` -- the readable family the ``kind`` folds into
    (``rate`` / ``field_solve`` / ``local_linear_map`` / ``matrix_free_map`` / ``projection`` /
    ``coupled_rate`` / ...). It is value-like: two handles compare equal iff their owner-qualified
    declaration identities match (the signature/category are DERIVED metadata), so a handle stays
    usable as a dict key or in tests. It carries no Program, no registry and no IR; the public
    ``P.call`` (and ADC-560's ``handle(...)`` facade) resolve the complete handle against the Program's
    bound registry. A same-named handle from another model is rejected.
    """

    __slots__ = ("signature", "category", "_registered_operator_name")

    @property
    def expression_readable(self) -> bool:
        """Operators are callable declarations, never scalar values."""
        return False

    @property
    def registered_operator_name(self) -> str:
        """Exact registry key this handle selects.

        Ordinary declarers use their local identity directly. Facade-specific
        subclasses may override this property when they expose a distinct
        presentation alias, but must carry that target explicitly; resolution
        never guesses from a signature or operator kind.
        """
        return self._registered_operator_name

    @property
    def qualified_id(self) -> str:
        """Complete public operator identity, including an explicit alias target."""
        return self._qualified_operator_id(self.owner_path)

    def _qualified_operator_id(self, owner_path: OwnerPath) -> str:
        return "%s::target::%s" % (
            self._qualified_id(owner_path),
            quote(self.registered_operator_name, safe=""),
        )

    def __init__(self, name: Any, *, kind: Any, owner: Any, signature: Any = None,
                 category: Any = None, schema_version: int = 1,
                 registered_operator_name: Any = None) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("OperatorHandle: name must be a non-empty string")
        if not isinstance(kind, str) or not kind:
            raise ValueError("OperatorHandle: kind must be a non-empty string")
        super().__init__(name, kind=kind, owner=owner, schema_version=schema_version)
        target = name if registered_operator_name is None else registered_operator_name
        if not isinstance(target, str) or not target:
            raise ValueError(
                "OperatorHandle: registered_operator_name must be a non-empty string")
        object.__setattr__(self, "_registered_operator_name", target)
        if signature is not None:
            from pops.model.signatures import Signature
            if not isinstance(signature, Signature):
                raise TypeError("OperatorHandle: signature must be a Signature or None")
        object.__setattr__(self, "signature", signature)
        if category is None:
            from pops.model.operators import operator_family
            category = operator_family(kind)
        if not isinstance(category, str) or not category:
            raise ValueError("OperatorHandle: category must be a non-empty string")
        object.__setattr__(self, "category", category)

    def _identity(self) -> tuple[Any, ...]:
        """Operator aliases with different registry targets are distinct values."""
        return super()._identity() + (self.registered_operator_name,)

    def canonical_identity(self) -> dict[str, Any]:
        """JSON-ready identity with both public alias and authenticated target."""
        owner = self.owner_path.canonical_declaration_path()
        result = super().canonical_identity()
        result["registered_operator_name"] = self.registered_operator_name
        result["qualified_id"] = self._qualified_operator_id(owner)
        return result

    def inspect(self) -> Any:
        """A structured, inert view of the operator this handle names (ADC-559).

        Returns a plain dict including ``name``, ``registered_operator_name``, ``kind``, ``category``
        and ``signature`` -- the readable identity a user
        or a report reads without touching the registry or the codegen. ``signature`` is the repr of
        the declared :class:`Signature` (``None`` when the declarer supplied none). Read-only: it
        touches no numerics, IR or Program.
        """
        view = super().inspect()
        view.update({"name": self.name, "category": self.category,
                     "signature": repr(self.signature) if self.signature is not None else None,
                     "registered_operator_name": self.registered_operator_name})
        return view

    def __call__(self, *args: Any, name: Any = None) -> Any:
        """Call the operator inside a time Program (ADC-560): the tableau-style facade over ``P.call``.

        Locates the Program from the first ProgramValue argument that carries a ``.prog`` back-reference and
        delegates to ``P.call(self, *args, name=name)`` -- the byte-identical
        lowering the public ``P.call(handle, ...)`` uses. So ``R(U, f)`` builds the SAME IR (same
        ``_ir_hash``), runs the SAME signature type-checks and raises the SAME signature errors as
        ``P.call(R, U, f)``, with ZERO numerics: ``__call__`` only builds IR. A call outside a Program
        (no ProgramValue argument to find the Program from) is refused with a clear error; the operator name
        stays an internal selector, never re-exposed as a public string.
        """
        prog = self._program_from_args(args)
        return prog.call(self, *args, name=name)

    def _program_from_args(self, args: Any) -> Any:
        """Find the time-Program to build IR into from the call arguments (ADC-560).

        The Program is the ``.prog`` back-reference on the first :class:`pops.time.values.ProgramValue`
        argument. A call with no such argument (outside a Program) is refused with a clear error
        naming the explicit ``P.call`` alternative. Shared by the base handle and its callable
        subtypes so the Program-location rule is defined once.
        """
        prog = next((a.prog for a in args if hasattr(a, "prog")), None)
        if prog is None:
            raise ValueError(
                "operator %r must be called with time-Program values (inside a Program) so it can "
                "find the Program to build IR into; got %r. Use P.call(%r, ...) if you hold the "
                "Program explicitly." % (self.name, args, self.name))
        return prog

    def __repr__(self) -> str:
        return "OperatorHandle(%r, kind=%r, owner=%r)" % (
            self.name, self.kind, self.owner_path.segments)


__all__ = ["Handle", "OperatorHandle", "OwnerPath"]
