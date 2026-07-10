"""Case-field, time-program, and parameter declaration registries."""
from __future__ import annotations

from typing import Any

from pops.model.ownership import MissingOwnershipError, OwnerKind, OwnerPath
from pops.problem._registry_freeze import (
    FreezableRegistry as _FreezableRegistry,
    flatten_freeze_members,
)
from pops.problem._registry_support import NO_KIND, strict_name
from pops.problem.handles import FieldHandle
from pops.problem.report import ProblemValidationReport


class FieldRegistry(_FreezableRegistry):
    """The elliptic field problems declared on a Problem (keyed on the field's name)."""

    family = "field"

    def __init__(self, owner: Any) -> None:
        self._owner_path = OwnerPath.coerce(owner).require_authoring_root(
            OwnerKind.CASE, where="FieldRegistry owner"
        )
        self._fields = {}
        self._handles = {}

    @property
    def owner_path(self) -> Any:
        return self._owner_path

    def _freezable_members(self) -> Any:
        return list(self._fields.values())

    def add(self, field_problem: Any) -> Any:
        """Register a FieldProblem exactly once and return its case-owned handle."""
        self._guard_frozen("add a field")
        from pops.fields import FieldProblem  # lazy: keep pops.problem free of a fields edge

        if not isinstance(field_problem, FieldProblem):
            raise TypeError(
                "field: expected a pops.fields.FieldProblem; got %r"
                % type(field_problem).__name__
            )
        key = strict_name(field_problem.name, "field name")
        if key in self._fields:
            raise ValueError("field: a field named %r already exists" % key)
        self._fields[key] = field_problem
        handle = FieldHandle(key, owner=self.owner_path, field_registry=self)
        self._handles[key] = handle
        return handle

    def handle(self, name: Any) -> FieldHandle:
        key = strict_name(name, "field name")
        try:
            return self._handles[key]
        except KeyError:
            raise KeyError(
                "unknown field %r (known: %s)"
                % (key, ", ".join(self._fields) or "<none>")
            ) from None

    def handles(self) -> Any:
        return dict(self._handles)

    def canonicalize(self, field: Any) -> FieldHandle:
        """Authenticate and resolve one case-owned field declaration."""
        from pops.model import Handle

        if not isinstance(field, Handle) or field.kind != "field":
            raise TypeError("field resolution requires a FieldHandle")
        registered = self._handles.get(field.local_id)
        if registered is None:
            raise MissingOwnershipError(
                "field handle %s is not registered by this case" % field.qualified_id
            )
        expected = registered._resolved(self.owner_path.canonical())
        object.__setattr__(expected, "_field_registry", None)
        matches = (
            field.canonical_identity() == expected.canonical_identity()
            if field.is_resolved
            else isinstance(field, FieldHandle) and field == registered
        )
        if not matches:
            raise MissingOwnershipError(
                "field handle %s is not registered by this case" % field.qualified_id
            )
        return expected

    def get(self, name: Any) -> Any:
        return self._fields.get(strict_name(name, "field name"))

    def names(self) -> Any:
        return list(self._fields)

    def items(self) -> Any:
        return self._fields.items()

    def resolved_items(self, resolver: Any) -> tuple[tuple[str, Any], ...]:
        """Return detached, reference-authenticated field declarations."""
        if not callable(resolver):
            raise TypeError("field declaration resolver must be callable")
        resolved = []
        for name, field in self._fields.items():
            protocol = getattr(field, "resolve_references", None)
            if not callable(protocol):
                raise TypeError(
                    "field %r must implement resolve_references(resolver)" % name)
            resolved.append((name, protocol(resolver)))
        return tuple(resolved)

    def solvers(self) -> Any:
        return {name: fp.solver for name, fp in self._fields.items() if fp.solver is not None}

    def __iter__(self) -> Any:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __contains__(self, name: Any) -> bool:
        return isinstance(name, str) and name in self._fields

    def validate(self, context: Any = None) -> Any:
        report = ProblemValidationReport()
        for name, field in self._fields.items():
            try:
                field.validate(context)
            except Exception as exc:  # noqa: BLE001 -- report the descriptor's own refusal
                report.error(self.family, "field_invalid", str(exc), context={"field": name})
        return report

    def inspect(self, resolver: Any = None) -> Any:
        items = (
            self.resolved_items(resolver) if callable(resolver) else self._fields.items()
        )
        return {name: field.inspect() for name, field in items}


class TimeRegistry(_FreezableRegistry):
    """The whole-system time Program slot."""

    family = "time"

    def __init__(self) -> None:
        self._program = None
        self._program_declared = False

    def _freezable_members(self) -> Any:
        return flatten_freeze_members(self._program)

    def set(self, program: Any) -> None:
        self._guard_frozen("set the time scheme")
        if self._program_declared:
            raise ValueError(
                "time program is already declared; resolve/read the existing declaration "
                "instead of registering it again"
            )
        self._program = program
        self._program_declared = True

    @property
    def program(self) -> Any:
        return self._program

    def names(self) -> Any:
        return [getattr(self._program, "name", "program")] if self._program is not None else []

    def __iter__(self) -> Any:
        return iter([self._program] if self._program is not None else [])

    def validate(self, context: Any = None) -> Any:
        return ProblemValidationReport()

    def inspect(self) -> Any:
        return {
            "program": getattr(self._program, "name", None)
            if self._program is not None
            else None
        }


class ParamRegistry(_FreezableRegistry):
    """The runtime and constant parameter declarations."""

    family = "params"

    def __init__(self) -> None:
        self._params = {}
        self._declarations = {}

    def _freezable_members(self) -> Any:
        return [declaration for declaration in self._declarations.values() if declaration is not None]

    def add(self, name: Any, default: Any = None, *, kind: Any = NO_KIND) -> None:
        self._guard_frozen("declare a param")
        if kind is not NO_KIND:
            raise TypeError(
                "param: the kind= string is removed (Spec 5 sec.7); pass a typed param object "
                "(pops.physics.RuntimeParam(name, value) or pops.physics.ConstParam(name, value)) "
                "instead of kind=%r" % (kind,)
            )
        if hasattr(name, "kind") and hasattr(name, "name") and hasattr(name, "value"):
            if default is not None:
                raise TypeError(
                    "param: a typed param was given; do not also pass a default (%r)" % (default,)
                )
            key = strict_name(name.name, "parameter name")
            spec = {"default": name.value, "kind": strict_name(name.kind, "parameter kind")}
            declaration = name
        elif getattr(name, "category", None) in ("runtime_param", "const_param") and hasattr(
            name, "name"
        ):
            if default is not None:
                raise TypeError(
                    "param: a typed param was given; do not also pass a default (%r)" % (default,)
                )
            kind_name = {"runtime_param": "runtime", "const_param": "const"}[name.category]
            declared = getattr(name, "default", getattr(name, "value", None))
            key = strict_name(name.name, "parameter name")
            spec = {"default": declared, "kind": kind_name}
            declaration = name
        else:
            key = strict_name(name, "parameter name")
            spec = {"default": default, "kind": "const"}
            declaration = None
        if key in self._params:
            raise ValueError(
                "parameter %r is already declared; parameter declarations are register-once" % key
            )
        self._params[key] = spec
        self._declarations[key] = declaration

    def get(self, name: Any) -> Any:
        return self._params.get(strict_name(name, "parameter name"))

    def names(self) -> Any:
        return list(self._params)

    def items(self) -> Any:
        return self._params.items()

    def declarations(self) -> Any:
        return dict(self._declarations)

    def __iter__(self) -> Any:
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def validate(self, context: Any = None) -> Any:
        return ProblemValidationReport()

    def inspect(self) -> Any:
        return {name: dict(spec) for name, spec in self._params.items()}


__all__ = ["FieldRegistry", "ParamRegistry", "TimeRegistry"]
