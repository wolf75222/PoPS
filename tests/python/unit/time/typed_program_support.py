"""Small typed authoring fixtures shared by the time-unit tests.

The production API accepts only ``Program.state(block[state])``. These helpers build real
Case/model declaration graphs so focused IR tests do not each need a page of assembly boilerplate.
They never wrap or weaken ``Program.state``: the final call always receives the exact qualified
instance handle.
"""
from __future__ import annotations

from typing import Any

from pops.model import (
    DeclarationIndex,
    Handle,
    OwnerKind,
    OwnerPath,
    OperatorRegistry,
    StateHandle,
    StateSpace,
)
from pops.problem import Case
from pops.time import Program


class _StateModel:
    def __init__(
        self,
        name: str,
        *,
        owner: OwnerPath | None = None,
        state_name: str = "U",
        space: StateSpace | None = None,
        registry: OperatorRegistry | None = None,
    ) -> None:
        self.name = name
        self.owner_path = owner or OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, name)
        # A typed declaration with zero components is not an "unknown" StateSpace; it is a real
        # zero-width state and must correctly fail a scalar/vector solve.  Unit fixtures that do not
        # provide a model therefore use the smallest honest state, one named component.
        declared_space = space or StateSpace(state_name, (state_name,))
        self.state = StateHandle(
            state_name, owner=self.owner_path, space=declared_space)
        self._operators = (
            registry if registry is not None else OperatorRegistry(owner=self.owner_path))

    def declaration_index(self) -> DeclarationIndex:
        return DeclarationIndex(owner=self.owner_path, handles=(self.state,))

    def operator_registry(self) -> OperatorRegistry:
        return self._operators


class _Context:
    def __init__(self, program: Program) -> None:
        self.program = program
        self.case = Case(name="typed-time-unit-case")
        self.models: dict[Any, _StateModel] = {}
        self.blocks: dict[tuple[str, str, Any], tuple[Any, Handle]] = {}
        self.fields: dict[str, Any] = {}


_CONTEXTS: dict[int, _Context] = {}


def _context(program: Program) -> _Context:
    if not isinstance(program, Program):
        raise TypeError("typed time fixture requires a Program")
    key = id(program)
    context = _CONTEXTS.get(key)
    if context is None or context.program is not program:
        context = _Context(program)
        _CONTEXTS[key] = context
    return context


def _space_key(space: StateSpace | None) -> Any:
    if space is None:
        return None
    if not isinstance(space, StateSpace):
        raise TypeError("typed state fixture space must be a StateSpace or None")
    return repr(space.to_data())


def _bound_owner(program: Program) -> Any:
    registries = getattr(program, "_operator_registries", {})
    if len(registries) != 1:
        return None
    return next(iter(registries))


def state_refs(
    program: Program,
    block_name: str = "fluid",
    *,
    state_name: str = "U",
    space: StateSpace | None = None,
    model: Any = None,
    state: Handle | None = None,
) -> tuple[Any, Handle]:
    """Return a real case-owned block and its model-local state declaration."""
    if not isinstance(block_name, str) or not block_name:
        raise TypeError("typed state fixture block_name must be a non-empty string")
    context = _context(program)
    selected_model = model
    selected_state = state

    if selected_model is None:
        requested_space = _space_key(space)
        for (known_block, known_state, _), existing in context.blocks.items():
            if known_block != block_name or known_state != state_name:
                continue
            declaration_space = _space_key(getattr(existing[1], "space", None))
            if space is None or declaration_space == requested_space:
                return existing
        for known_key, candidate in context.models.items():
            candidate_state = candidate.state
            if candidate_state.local_id != state_name:
                continue
            candidate_space = _space_key(getattr(candidate_state, "space", None))
            if space is None or candidate_space == requested_space:
                selected_model = candidate
                selected_state = candidate_state
                break

    if selected_model is not None:
        if selected_state is None:
            declarations = tuple(selected_model.declaration_index().records())
            candidates = [item for item in declarations if item.kind == "state"]
            if len(candidates) != 1:
                raise ValueError(
                    "typed state fixture model must expose exactly one state declaration"
                )
            selected_state = candidates[0]
        selected_space = space
        if selected_space is None:
            selected_space = getattr(selected_state, "space", None)
        model_key: Any = ("explicit", id(selected_model))
        context.models.setdefault(model_key, selected_model)
    else:
        owner = _bound_owner(program)
        bound_registry = None
        if owner is not None:
            bound_registry = program._operator_registries[owner]
        selected_space = space
        model_key = ("proxy", owner, state_name, _space_key(selected_space))
        selected_model = context.models.get(model_key)
        if selected_model is None:
            selected_model = _StateModel(
                owner.name if owner is not None else "typed_time_model_%s" % state_name,
                owner=owner,
                state_name=state_name,
                space=selected_space,
                registry=bound_registry,
            )
            context.models[model_key] = selected_model
        selected_state = selected_model.state

    block_key = (block_name, state_name, model_key)
    existing = context.blocks.get(block_key)
    if existing is not None:
        return existing
    block = context.case.block(block_name, selected_model, states=(selected_state,))
    result = (block, selected_state)
    context.blocks[block_key] = result
    return result


def typed_state(
    program: Program,
    block_name: str = "fluid",
    *,
    state_name: str | None = None,
    space: StateSpace | None = None,
    model: Any = None,
    state: Handle | None = None,
) -> Any:
    """Declare through the final two-handle API and return the requested public view.

    Migrated one-argument authoring sites omit ``state_name`` and receive the readable ``.n``
    ProgramValue. Sites which explicitly name the state receive its TimeState declaration so they
    can use ``.next``/``.stage``/``.prev``. Both routes call the exact same typed production API.
    """
    declaration_name = state_name or (space.name if space is not None else "U")
    block, declaration = state_refs(
        program,
        block_name,
        state_name=declaration_name,
        space=space,
        model=model,
        state=state,
    )
    temporal = program.state(block[declaration])
    return temporal if state_name is not None else temporal.n


def fresh_state_refs(
    block_name: str = "fluid",
    *,
    state_name: str = "U",
    space: StateSpace | None = None,
) -> tuple[Any, Handle]:
    """Build typed arguments for a preset that creates its own Program."""
    holder = Program("typed-preset-reference-holder")
    return state_refs(holder, block_name, state_name=state_name, space=space)


def fresh_field_refs(
    model: Any,
    *,
    block_name: str = "fluid",
    field_name: str = "potential",
    provider: Any = None,
) -> tuple[Any, Any]:
    """Build one exact factory state reference and its Case-owned field solve authority."""
    module = getattr(model, "module", model)
    declarations = tuple(
        item for item in model.declaration_index().records() if item.kind == "state")
    if len(declarations) != 1:
        raise ValueError("fresh field fixture requires exactly one state declaration")
    state = declarations[0]
    case = Case(name="typed-field-factory-case")
    block = case.block(block_name, model, states=(state,))

    from pops.descriptors import Descriptor
    from pops.fields import FieldDiscretization, FieldOperator
    from pops.ir import ValueExpr
    from pops.math import laplacian
    from pops.model import Module, Signature

    provider_space = getattr(getattr(provider, "signature", None), "output", None)
    components = (
        tuple(provider_space.components)
        if provider_space is not None else (field_name,)
    )
    field_module = Module("typed_factory_field_%s" % field_name)
    field_space = field_module.field_space(field_name, components)
    if provider is None:
        provider = field_module.operator(
            name=field_name + "_provider",
            signature=Signature((state.space,), field_space),
            kind="field_operator",
            expr=field_name + "_provider",
        )
    field_block = case.block("typed_field_%s" % field_name, field_module)
    unknown = field_block[field_module.field_handle(field_space)]

    class _Method(Descriptor):
        category = "field_method"

        def to_data(self) -> dict[str, Any]:
            return {"type": "unit-second-order"}

    class _Solver(Descriptor):
        category = "elliptic_solver"

        def to_data(self) -> dict[str, Any]:
            return {"type": "unit-krylov"}

    operator = FieldOperator(
        field_name,
        unknown=unknown,
        equation=-laplacian(ValueExpr(unknown)) == ValueExpr(block[state]),
        providers=provider,
    )
    field = case.field(
        operator,
        FieldDiscretization(method=_Method(), boundaries=(), solver=_Solver()),
    )
    return block[state], field


def typed_field(program: Program, name: str, *, provider: Any = None) -> Any:
    """Return the callable Case-owned ``FieldHandle`` for the final field-solve route."""
    if not isinstance(name, str) or not name:
        raise TypeError("typed field fixture name must be a non-empty string")
    context = _context(program)
    existing = context.fields.get(name)
    if existing is not None:
        return existing
    from pops.descriptors import Descriptor
    from pops.fields import FieldDiscretization, FieldOperator
    from pops.ir import ValueExpr
    from pops.math import laplacian
    from pops.model import Module, Signature

    if not context.models:
        state_refs(program)
    state_block, state_declaration = next(iter(context.blocks.values()))
    state_ref = state_block[state_declaration]
    field_module = Module("typed_time_field_%s" % name)
    provider_space = getattr(getattr(provider, "signature", None), "output", None)
    components = (
        tuple(provider_space.components)
        if provider_space is not None else (name,)
    )
    field_space = field_module.field_space(name, components)
    if provider is None:
        provider = field_module.operator(
            name=name + "_provider",
            signature=Signature((state_declaration.space,), field_space),
            kind="field_operator",
            expr=name + "_provider",
        )
    field_block = context.case.block("typed_field_%s" % name, field_module)
    unknown = field_block[field_module.field_handle(field_space)]

    class _Method(Descriptor):
        category = "field_method"

        def to_data(self) -> dict[str, Any]:
            return {"type": "unit-second-order"}

    class _Solver(Descriptor):
        category = "elliptic_solver"

        def to_data(self) -> dict[str, Any]:
            return {"type": "unit-krylov"}

    operator = FieldOperator(
        name,
        unknown=unknown,
        equation=-laplacian(ValueExpr(unknown)) == ValueExpr(state_ref),
        providers=provider,
    )
    discretization = FieldDiscretization(
        method=_Method(), boundaries=(), solver=_Solver())
    field = context.case.field(operator, discretization)
    context.fields[name] = field
    return field


def solve_field(
    program: Program,
    state: Any,
    *,
    field: Any = None,
    name: str | None = None,
    action: Any = None,
) -> Any:
    """Evaluate one callable field handle and consume its fallible outcome.

    This is test assembly, not a compatibility method on ``Program``: every call crosses the final
    Case-owned ``FieldHandle`` boundary and records explicit failure handling.
    """
    from pops.time import FailRun

    selected = typed_field(program, "potential") if field is None else field
    outcome = selected(state, name=name)
    return outcome.consume(action=FailRun() if action is None else action)


def solve_field_blocks(
    program: Program,
    states: Any,
    *,
    field: Any = None,
    name: str | None = None,
    action: Any = None,
) -> Any:
    """Evaluate one authenticated coupled field handle over exact same-stage states."""
    from pops.time import FailRun

    values = tuple(states)
    selected = typed_field(program, "potential") if field is None else field
    outcome = selected(*values, name=name)
    return outcome.consume(action=FailRun() if action is None else action)


def commits_by_block(program: Program) -> dict[str, Any]:
    """Readable test projection of typed commit keys onto their block labels."""
    return {
        state.block_ref.local_id: value
        for state, value in program.commits().items()
    }


__all__ = [
    "commits_by_block", "fresh_field_refs", "fresh_state_refs", "solve_field",
    "solve_field_blocks", "state_refs", "typed_field", "typed_state",
]
