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
    rate = R(U, fields)              # the handle is the sole public selector

``OperatorHandle.__call__`` locates the Program from its typed values and delegates to the one
private lowering seam. ``R(U, f)`` therefore runs exact owner/kind/signature checks and builds IR
with zero numerics; a bare operator-name string is never a public selector.
The handle holds no Program reference, but it works only in a Program bound to its exact declaring
owner; a homonymous registry from another model is rejected.

This module imports only the standard library (and the sibling ``pops.model`` types lazily inside
methods) so it stays codegen-free and ``_pops``-free and keeps the ``pops.time`` import graph acyclic
(``pops.time`` already imports ``pops.model``; ``pops.model`` never imports ``pops.time``).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from .ownership import OwnerPath, UnresolvedOwnershipError

if TYPE_CHECKING:
    from .spaces import StateSpace


_KEEP_REFERENCE = object()

class Handle:
    """Immutable, owner-qualified identity of one declared object."""

    __slots__ = (
        "owner_path", "local_id", "kind", "schema_version",
        "_declaration_ref", "_block_ref",
    )
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
        object.__setattr__(self, "_declaration_ref", None)
        object.__setattr__(self, "_block_ref", None)

    @property
    def name(self) -> str:
        return self.local_id

    @property
    def expression_readable(self) -> bool:
        """Whether an explicit ValueExpr may read this declaration as a symbolic value."""
        return True

    @property
    def is_resolved(self) -> bool:
        """Whether this handle carries canonical, serialisable ownership."""
        return self.owner_path.is_canonical

    @property
    def is_instance(self) -> bool:
        """Whether this is a block-qualified projection of a model declaration."""
        return self._declaration_ref is not None

    @property
    def declaration_ref(self) -> Handle | None:
        """Original scientific declaration for a block-qualified handle."""
        return self._declaration_ref

    @property
    def block_ref(self) -> Handle | None:
        """Block declaration selecting this instance, when qualified."""
        return self._block_ref

    def _instance_refs(self) -> tuple[Handle, Handle]:
        declaration = self.declaration_ref
        block = self.block_ref
        if declaration is None or block is None:
            raise RuntimeError("instance Handle is missing its declaration/block references")
        return declaration, block

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
        # Inspection removes the opaque authoring capability but is not an authentication event.
        # A newly authored model may not have built its declaration index yet, so use the explicit
        # presentation projection rather than inventing or requiring a canonical fingerprint.
        inspection_owner = self.owner_path.presentation()
        result = {
            "kind": self.kind,
            "local_id": self.local_id,
            "owner_path": inspection_owner.to_data(),
            "ownership_phase": "canonical" if self.is_resolved else "authoring",
            "qualified_id": self._qualified_id(inspection_owner),
            "schema_version": self.schema_version,
        }
        if self.is_instance:
            declaration, block = self._instance_refs()
            result["declaration_ref"] = declaration.inspect()
            result["block_ref"] = block.inspect()
        return result

    def canonical_identity(self) -> dict[str, Any]:
        """JSON-ready declaration identity with its complete typed owner path."""
        owner = self.owner_path
        if owner.is_authoring:
            raise UnresolvedOwnershipError(
                "handle %s is still authoring-owned; resolve it through its authoritative "
                "registry before canonical serialization" % self.qualified_id)
        result = {
            "kind": self.kind,
            "local_id": self.local_id,
            "owner_path": owner.to_data(),
            "qualified_id": self._qualified_id(owner),
            "schema_version": self.schema_version,
        }
        if self.is_instance:
            declaration, block = self._instance_refs()
            if not declaration.is_resolved or not block.is_resolved:
                raise UnresolvedOwnershipError(
                    "instance handle %s retains unresolved declaration/block references"
                    % self.qualified_id)
            result["declaration_ref"] = declaration.canonical_identity()
            result["block_ref"] = block.canonical_identity()
        return result

    @classmethod
    def from_canonical_identity(cls, data: Any) -> Handle:
        """Rebuild and authenticate the identity emitted by :meth:`canonical_identity`."""
        if not isinstance(data, Mapping):
            raise TypeError("Handle canonical identity must be a mapping")
        required = {"kind", "local_id", "owner_path", "qualified_id", "schema_version"}
        operator_keys = required | {"registered_operator_name"}
        block_keys = required | {"handle_type", "model_owner_path"}
        parameter_keys = required | {"handle_type", "param_kind"}
        instance_keys = required | {"declaration_ref", "block_ref"}
        operator_instance_keys = operator_keys | {"declaration_ref", "block_ref"}
        parameter_instance_keys = parameter_keys | {"declaration_ref", "block_ref"}
        allowed_shapes = (
            required,
            operator_keys,
            block_keys,
            parameter_keys,
            instance_keys,
            operator_instance_keys,
            parameter_instance_keys,
        )
        if set(data) not in allowed_shapes:
            raise TypeError(
                "Handle canonical identity has an unsupported key set %s"
                % sorted(data))
        owner = OwnerPath.from_data(data["owner_path"])
        if data.get("handle_type") == "block":
            if data["kind"] != "block":
                raise ValueError(
                    "BlockHandle canonical identity requires handle_type='block' and kind='block'"
                )
            from pops.problem.handles import BlockHandle

            result = BlockHandle(
                data["local_id"],
                owner=owner,
                model_owner=OwnerPath.from_data(data["model_owner_path"]),
                schema_version=data["schema_version"],
            )
        elif data.get("handle_type") == "parameter":
            if data["kind"] != "parameter":
                raise ValueError(
                    "ParamHandle canonical identity requires handle_type='parameter' and "
                    "kind='parameter'"
                )
            result = ParamHandle(
                data["local_id"], owner=owner, param_kind=data["param_kind"],
                schema_version=data["schema_version"])
        elif "handle_type" in data:
            raise ValueError("unknown Handle handle_type %r" % data["handle_type"])
        elif "registered_operator_name" in data:
            result: Handle = OperatorHandle(
                data["local_id"], kind=data["kind"], owner=owner,
                schema_version=data["schema_version"],
                registered_operator_name=data["registered_operator_name"])
        else:
            result = Handle(
                data["local_id"], kind=data["kind"], owner=owner,
                schema_version=data["schema_version"])
        if "declaration_ref" in data:
            declaration_ref = cls.from_canonical_identity(data["declaration_ref"])
            block_ref = cls.from_canonical_identity(data["block_ref"])
            object.__setattr__(result, "_declaration_ref", declaration_ref)
            object.__setattr__(result, "_block_ref", block_ref)
        if result.canonical_identity() != dict(data):
            raise ValueError("Handle canonical identity has an invalid qualified_id or payload")
        return result

    def _with_owner(
        self,
        owner: Any,
        *,
        declaration_ref: Any = _KEEP_REFERENCE,
        block_ref: Any = _KEEP_REFERENCE,
    ) -> Handle:
        """Internal immutable requalification preserving the concrete handle metadata."""
        qualified_owner = OwnerPath.coerce(owner)
        clone = object.__new__(type(self))
        for base in reversed(type(self).__mro__):
            slots = base.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot in ("__dict__", "__weakref__") or not hasattr(self, slot):
                    continue
                object.__setattr__(clone, slot, getattr(self, slot))
        object.__setattr__(clone, "owner_path", qualified_owner)
        if declaration_ref is not _KEEP_REFERENCE:
            if declaration_ref is not None and not isinstance(declaration_ref, Handle):
                raise TypeError("declaration_ref must be a Handle or None")
            object.__setattr__(clone, "_declaration_ref", declaration_ref)
        if block_ref is not _KEEP_REFERENCE:
            if block_ref is not None and not isinstance(block_ref, Handle):
                raise TypeError("block_ref must be a Handle or None")
            object.__setattr__(clone, "_block_ref", block_ref)
        return clone

    def _resolved(self, owner: Any = None) -> Handle:
        """Return a canonical copy; authoritative registries call this after authentication."""
        resolved_owner = (self.owner_path.canonical() if owner is None
                          else OwnerPath.coerce(owner))
        if resolved_owner.is_authoring:
            raise UnresolvedOwnershipError("resolved handle owner must be canonical")
        declaration_ref = (self.declaration_ref._resolved()
                           if self.declaration_ref is not None else None)
        block_ref = self.block_ref._resolved() if self.block_ref is not None else None
        return self._with_owner(
            resolved_owner, declaration_ref=declaration_ref, block_ref=block_ref)

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
            type(self).__name__, self.local_id, self.kind, str(self.owner_path))


class StateHandle(Handle):
    """Registry-issued state identity carrying its authoritative :class:`StateSpace`.

    The space is declaration metadata, not a second identity axis. Carrying it on the handle lets
    ``Program.state(block[state])`` stay typed without consulting a live Module registry.
    """

    __slots__ = ("space",)

    space: StateSpace

    def __init__(self, name: Any, *, owner: Any, space: Any, schema_version: int = 1) -> None:
        from .spaces import StateSpace

        if not isinstance(space, StateSpace):
            raise TypeError("StateHandle space must be a pops.model.StateSpace")
        if name != space.name:
            raise ValueError(
                "StateHandle name %r must match StateSpace name %r" % (name, space.name))
        super().__init__(name, kind="state", owner=owner, schema_version=schema_version)
        object.__setattr__(self, "space", space)

    def inspect(self) -> dict[str, Any]:
        result = super().inspect()
        result["space"] = self.space.to_data()
        return result


class ParamHandle(Handle):
    """Immutable identity of one canonical parameter declaration.

    A ParamHandle deliberately exposes no arithmetic operators.  Symbolic parameter reads are
    separate Expr nodes; the handle remains a stable dictionary key with Boolean equality.
    """

    __slots__ = ("param_kind",)
    _PARAM_KINDS = frozenset({"runtime", "const", "derived"})

    def __init__(
        self,
        name: Any,
        *,
        owner: Any,
        param_kind: Any,
        schema_version: int = 1,
    ) -> None:
        value = getattr(param_kind, "value", param_kind)
        if value not in self._PARAM_KINDS:
            raise ValueError(
                "ParamHandle param_kind must be runtime, const or derived (got %r)" % value
            )
        super().__init__(
            name, kind="parameter", owner=owner, schema_version=schema_version
        )
        object.__setattr__(self, "param_kind", value)

    @property
    def qualified_id(self) -> str:
        return self._qualified_param_id(self.owner_path)

    def _qualified_param_id(self, owner_path: OwnerPath) -> str:
        return "%s::param-kind::%s" % (
            self._qualified_id(owner_path), quote(self.param_kind, safe="")
        )

    def inspect(self) -> dict[str, Any]:
        result = super().inspect()
        result.update({
            "handle_type": "parameter",
            "param_kind": self.param_kind,
            "qualified_id": self._qualified_param_id(self.owner_path.presentation()),
        })
        return result

    def canonical_identity(self) -> dict[str, Any]:
        result = super().canonical_identity()
        result.update({
            "handle_type": "parameter",
            "param_kind": self.param_kind,
            "qualified_id": self._qualified_param_id(self.owner_path),
        })
        return result

    def _identity(self) -> tuple[Any, ...]:
        return super()._identity() + (self.param_kind,)


class OperatorHandle(Handle):
    """A typed, INSPECTABLE reference to a declared operator (Spec 5 sec.14.2.3, ADC-559).

    Carries the operator ``name``, an optional ``kind`` (the operator-first kind, e.g.
    ``"local_rate"`` / ``"local_source"`` / ``"local_linear_operator"`` / ``"field_operator"``),
    the declared :class:`pops.model.signatures.Signature` (or ``None`` when the declarer did not
    supply one) and the mathematical ``category`` -- the readable family the ``kind`` folds into
    (``rate`` / ``field_solve`` / ``local_linear_map`` / ``matrix_free_map`` / ``projection`` /
    ``coupled_rate`` / ...). It is value-like: two handles compare equal iff their owner-qualified
    declaration identities match (the signature/category are DERIVED metadata), so a handle stays
    usable as a dict key or in tests. It carries no Program, no registry and no IR; calling the
    handle resolves it against the Program's bound registry. A same-named handle from another model
    is rejected.
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
        owner = self.owner_path
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

    def __call__(
        self,
        *args: Any,
        name: Any = None,
        schedule: Any = None,
        program: Any = None,
    ) -> Any:
        """Call the operator inside a time Program (ADC-560).

        ``R(U, f)`` locates its Program from the first ProgramValue argument. A genuinely nullary
        operator has no such value, so its sole form is ``L(program=T)``. ``program=`` is refused
        when positional arguments exist: there is always one Program authority, never a redundant
        selector. Both forms delegate to the same typed lowering and retain this exact handle.
        """
        if program is None:
            prog = self._program_from_args(args)
        else:
            if args:
                raise TypeError(
                    "operator %r accepts program= only for a nullary call; ProgramValue "
                    "arguments already select their Program" % self.name)
            from pops.time._program.api import Program

            if not isinstance(program, Program):
                raise TypeError(
                    "operator %r program= must be a pops.Program, got %r"
                    % (self.name, program))
            prog = program
        return prog._call(self, *args, name=name, schedule=schedule)

    def _program_from_args(self, args: Any) -> Any:
        """Find the time-Program to build IR into from the call arguments (ADC-560).

        The Program is the ``.prog`` back-reference on the first :class:`pops.time.values.ProgramValue`
        argument. A call with no such argument is refused and directs nullary operators to the
        explicit ``operator(program=T)`` form. Shared by the base handle and its callable subtypes
        so the Program-location rule is defined once.
        """
        prog = next((a.prog for a in args if hasattr(a, "prog")), None)
        if prog is None:
            raise ValueError(
                "operator %r must be called with time-Program values (inside a Program) so it can "
                "find the Program to build IR into; a nullary operator uses "
                "operator(program=T); got %r." % (self.name, args))
        return prog

    def __repr__(self) -> str:
        return "OperatorHandle(%r, kind=%r, owner=%r)" % (
            self.name, self.kind, str(self.owner_path))


__all__ = ["Handle", "ParamHandle", "OperatorHandle", "OwnerPath"]
