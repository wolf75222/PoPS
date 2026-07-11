"""Owner-aware registry for canonical parameter declarations."""
from __future__ import annotations

from typing import Any

from pops.params.runtime import (
    DerivedParam,
    ParamKind,
    ParamPhase,
    ParameterDeclaration,
)

from .handles import ParamHandle
from .ownership import OwnerKind, OwnerPath
from .registry import DeclarationIndex


_DERIVED_PHASE_DEPENDENCIES = {
    ParamPhase.Compile: frozenset({ParamPhase.Compile}),
    ParamPhase.Bind: frozenset({ParamPhase.Compile, ParamPhase.Bind}),
    ParamPhase.Runtime: frozenset({ParamPhase.Compile, ParamPhase.Bind, ParamPhase.Runtime}),
    ParamPhase.PerBlock: frozenset(
        {ParamPhase.Compile, ParamPhase.Bind, ParamPhase.Runtime, ParamPhase.PerBlock}
    ),
    ParamPhase.PerLevel: frozenset(
        {ParamPhase.Compile, ParamPhase.Bind, ParamPhase.Runtime, ParamPhase.PerLevel}
    ),
}


class ParamRegistry:
    """Register-once parameter authority for one model definition."""

    __slots__ = (
        "_owner_path", "_authority_token", "_declarations", "_handles", "_mutation_guard",
    )

    def __init__(self, *, owner: Any, mutation_guard: Any = None) -> None:
        candidate = OwnerPath.coerce(owner)
        if candidate.kind not in {OwnerKind.MODEL_DEFINITION, OwnerKind.CASE}:
            raise ValueError(
                "ParamRegistry owner must be a MODEL_DEFINITION or CASE authority, got %s"
                % candidate
            )
        self._owner_path = candidate.require_authoring_root(
            candidate.kind, where="ParamRegistry owner"
        )
        self._authority_token = object()
        self._declarations: dict[str, ParameterDeclaration] = {}
        self._handles: dict[str, ParamHandle] = {}
        if mutation_guard is not None and not callable(mutation_guard):
            raise TypeError("ParamRegistry mutation_guard must be callable or None")
        self._mutation_guard = mutation_guard

    @property
    def owner_path(self) -> OwnerPath:
        return self._owner_path

    def register(self, declaration: Any) -> ParamHandle:
        if self._mutation_guard is not None:
            self._mutation_guard("register a parameter")
        if not isinstance(declaration, ParameterDeclaration):
            raise TypeError(
                "Module.param requires RuntimeParam, ConstParam or DerivedParam; got %s"
                % type(declaration).__name__
            )
        name = declaration.name
        if name in self._declarations:
            raise ValueError(
                "parameter %r is already declared; parameter declarations are register-once"
                % name
            )
        if isinstance(declaration, DerivedParam):
            self._validate_derived_dependencies(declaration)
        declaration.validate()
        if isinstance(declaration, DerivedParam) and declaration.phase is ParamPhase.Compile:
            from pops.ir.param_values import (
                _evaluate_compile_derived,
                _validate_derived_result,
            )

            resolved = _evaluate_compile_derived(self, declaration, stack=())
            _validate_derived_result(declaration, resolved)
            declaration._set_compile_resolved_value(resolved)
        handle = ParamHandle(
            name, owner=self.owner_path, param_kind=declaration.kind
        )
        declaration._claim_owner(self._authority_token, str(self.owner_path))
        self._declarations[name] = declaration
        self._handles[name] = handle
        return handle

    def _validate_derived_dependencies(self, declaration: DerivedParam) -> None:
        declared_by_local = {}
        declared_by_qid = {}
        for dependency in declaration.depends_on:
            registered = self._handles.get(dependency.local_id)
            if registered is None or registered != dependency:
                raise ValueError(
                    "DerivedParam %r dependency %s is not issued by this Module ParamRegistry"
                    % (declaration.name, dependency.qualified_id)
                )
            dependency_decl = self._declarations[dependency.local_id]
            if dependency_decl.kind is ParamKind.Runtime and declaration.phase is ParamPhase.Compile:
                raise ValueError(
                    "DerivedParam %r at Compile cannot depend on runtime parameter %s"
                    % (declaration.name, dependency.qualified_id)
                )
            if dependency_decl.kind is ParamKind.Derived:
                allowed = _DERIVED_PHASE_DEPENDENCIES[declaration.phase]
                if dependency_decl.phase not in allowed:
                    raise ValueError(
                        "DerivedParam %r phase %s cannot depend on %s phase %s"
                        % (
                            declaration.name,
                            declaration.phase.value,
                            dependency.qualified_id,
                            dependency_decl.phase.value,
                        )
                    )
            declared_by_local[dependency.local_id] = dependency
            declared_by_qid[dependency.qualified_id] = dependency

        from ._bind_expression import expression_reference_keys

        payload = declaration._expression_payload
        references = expression_reference_keys(
            payload["value"], where="DerivedParam %r" % declaration.name
        )
        for reference_kind, reference in references:
            dependency = (declared_by_local.get(reference) if reference_kind == "local"
                          else declared_by_qid.get(reference))
            if dependency is None:
                raise ValueError(
                    "DerivedParam %r expression reads undeclared dependency %r"
                    % (declaration.name, reference)
                )
        live_references = set(declaration.expression.declaration_references())
        declared_references = set(declaration.depends_on)
        foreign = live_references - declared_references
        if foreign:
            raise ValueError(
                "DerivedParam %r expression carries foreign parameter handle(s): %s"
                % (declaration.name, ", ".join(sorted(
                    reference.qualified_id for reference in foreign)))
            )
        unused = [dependency.qualified_id for dependency in declaration.depends_on
                  if dependency not in live_references]
        if unused:
            raise ValueError(
                "DerivedParam %r depends_on contains unread parameter(s): %s"
                % (declaration.name, ", ".join(unused))
            )

    def handle(self, declaration_or_handle: Any) -> ParamHandle:
        if isinstance(declaration_or_handle, str):
            raise TypeError(
                "parameter handle lookup requires the declaration or its ParamHandle, not a string"
            )
        if isinstance(declaration_or_handle, ParamHandle):
            registered = self._handles.get(declaration_or_handle.local_id)
            if registered is None:
                raise KeyError("unknown parameter %r" % declaration_or_handle.local_id)
            if registered != declaration_or_handle:
                raise ValueError(
                    "parameter handle %s belongs to another Module registry"
                    % declaration_or_handle.qualified_id
                )
            return registered
        if not isinstance(declaration_or_handle, ParameterDeclaration):
            raise TypeError("parameter handle lookup requires a ParameterDeclaration")
        registered_decl = self._declarations.get(declaration_or_handle.name)
        if registered_decl is None:
            raise KeyError("unknown parameter %r" % declaration_or_handle.name)
        if registered_decl is not declaration_or_handle:
            raise ValueError(
                "parameter %r belongs to another Module registry"
                % declaration_or_handle.name
            )
        return self._handles[declaration_or_handle.name]

    def declaration(self, handle: Any) -> ParameterDeclaration:
        authenticated = self.handle(handle)
        return self._declarations[authenticated.local_id]

    def declarations(self) -> dict[str, ParameterDeclaration]:
        return dict(self._declarations)

    def handles(self) -> tuple[ParamHandle, ...]:
        return tuple(self._handles.values())

    def items(self) -> Any:
        return self._declarations.items()

    def declaration_index(self) -> DeclarationIndex:
        return DeclarationIndex(owner=self.owner_path, handles=self.handles())


__all__ = ["ParamRegistry"]
