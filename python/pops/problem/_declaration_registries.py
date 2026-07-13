"""Case-field, time-program, and parameter declaration registries."""
from __future__ import annotations

from typing import Any

from pops.descriptors import Availability, Descriptor
from pops.model.ownership import MissingOwnershipError, OwnerKind, OwnerPath
from pops.model.param_registry import ParamRegistry as _CanonicalParamRegistry
from pops.problem._registry_freeze import (
    FreezableRegistry as _FreezableRegistry,
    flatten_freeze_members,
)
from pops.problem._registry_support import strict_name
from pops.problem.handles import FieldHandle
from pops._report import ReportTree


def _validation_root(source: str) -> ReportTree:
    return ReportTree(
        phase="validation", severity="info", code="validation.%s.root" % source,
        source=source)


class FieldRegistry(_FreezableRegistry):
    """Case-owned bindings from physical field operators to numerical plans."""

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

    def add(self, operator: Any, discretization: Any) -> Any:
        """Register exactly one ``FieldOperator + FieldDiscretization`` pair."""
        self._guard_frozen("add a field")
        from pops.fields import FieldDiscretization, FieldOperator

        if not isinstance(operator, FieldOperator):
            raise TypeError(
                "field: operator must be a pops.fields.FieldOperator; got %r"
                % type(operator).__name__
            )
        if not isinstance(discretization, FieldDiscretization):
            raise TypeError(
                "field: discretization must be a pops.fields.FieldDiscretization; got %r"
                % type(discretization).__name__
            )
        key = strict_name(operator.name, "field operator name")
        if key in self._fields:
            raise ValueError("field: a field operator named %r already exists" % key)
        self._fields[key] = _RegisteredField(operator, discretization)
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

    def __iter__(self) -> Any:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __contains__(self, name: Any) -> bool:
        return isinstance(name, str) and name in self._fields

    def validate(self, context: Any = None) -> Any:
        report = _validation_root(self.family)
        for name, field in self._fields.items():
            try:
                field.validate(context)
            except Exception as exc:  # noqa: BLE001 -- report the descriptor's own refusal
                report = report.error(
                    self.family, "field_invalid", str(exc), context={"field": name})
        return report

    def inspect(self, resolver: Any = None) -> Any:
        items = (
            self.resolved_items(resolver) if callable(resolver) else self._fields.items()
        )
        return {name: field.inspect() for name, field in items}


class _RegisteredField(Descriptor):
    """Internal case binding; users author only its two public descriptor inputs."""

    category = "registered_field"

    def __init__(self, operator: Any, discretization: Any) -> None:
        self.operator = operator
        self.discretization = discretization

    @property
    def name(self) -> str:
        return self.operator.name

    def options(self) -> dict[str, Any]:
        return {"operator": self.operator, "discretization": self.discretization}

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "operator": self.operator.to_data(),
            "discretization": self.discretization.to_data(),
        }

    def available(self, context: Any = None) -> Availability:
        for value in (self.operator, self.discretization):
            status = value.available(context)
            if not status.ok:
                return status
        return Availability.yes("field operator and discretization are available")

    def validate(self, context: Any = None) -> bool:
        self.operator.validate(context)
        self.discretization.validate(context)
        return True

    def resolve_references(self, resolver: Any) -> Any:
        return type(self)(
            self.operator.resolve_references(resolver),
            self.discretization.resolve_references(resolver),
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "operator": self.operator.inspect(),
            "discretization": self.discretization.inspect(),
        }


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
        return _validation_root(self.family)

    def inspect(self) -> Any:
        return {
            "program": getattr(self._program, "name", None)
            if self._program is not None
            else None
        }


class ParamRegistry(_CanonicalParamRegistry, _FreezableRegistry):
    """Canonical case-owned parameter authority.

    This is the same owner-qualified registry used by ``pops.model.Module``.
    The Problem wrapper adds only the freeze/report protocol; it does not keep a
    second flat ``{name: value}`` store.
    """

    family = "params"

    def __init__(self, owner: Any) -> None:
        _CanonicalParamRegistry.__init__(self, owner=owner)
        self._frozen = False

    def add(self, declaration: Any) -> Any:
        self._guard_frozen("declare a parameter")
        return self.register(declaration)

    def get(self, parameter: Any) -> Any:
        if isinstance(parameter, str):
            return self._declarations.get(strict_name(parameter, "parameter name"))
        return self.declaration(parameter)

    def names(self) -> Any:
        return list(self._declarations)

    def canonicalize(self, parameter: Any) -> Any:
        authenticated = self.handle(parameter)
        return authenticated._resolved(self.owner_path.canonical())

    def _freezable_members(self) -> Any:
        return list(self._declarations.values())

    def freeze(self) -> Any:
        if self._frozen:
            return self
        from types import MappingProxyType

        for declaration in self._declarations.values():
            declaration.freeze()
        object.__setattr__(self, "_declarations", MappingProxyType(dict(self._declarations)))
        object.__setattr__(self, "_handles", MappingProxyType(dict(self._handles)))
        object.__setattr__(self, "_frozen", True)
        return self

    def __iter__(self) -> Any:
        return iter(self._declarations)

    def __len__(self) -> int:
        return len(self._declarations)

    def __contains__(self, name: Any) -> bool:
        return isinstance(name, str) and name in self._declarations

    def validate(self, context: Any = None) -> Any:
        report = _validation_root(self.family)
        for name, declaration in self._declarations.items():
            try:
                declaration.validate(context)
            except Exception as exc:  # noqa: BLE001 - aggregate descriptor refusal
                report = report.error(
                    self.family,
                    "parameter_invalid",
                    str(exc),
                    context={"parameter": name},
                )
        return report

    def inspect(self) -> Any:
        return {
            name: {
                **declaration.bind_data(),
                "handle": self._handles[name].inspect(),
            }
            for name, declaration in self._declarations.items()
        }


__all__ = ["FieldRegistry", "ParamRegistry", "TimeRegistry"]
