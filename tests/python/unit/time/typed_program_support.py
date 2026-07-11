"""Small typed authoring fixtures shared by the time-unit tests.

The production API accepts only ``Program.state(BlockHandle, state Handle)``.  These helpers build
real Problem/model declaration graphs so focused IR tests do not each need a page of assembly
boilerplate.  They never wrap or weaken ``Program.state``: the final call always receives the two
typed semantic handles.
"""
from __future__ import annotations

from typing import Any

from pops.model import (
    DeclarationIndex,
    Handle,
    OwnerKind,
    OwnerPath,
    StateHandle,
    StateSpace,
)
from pops.problem import Problem
from pops.time import Program


class _StateModel:
    def __init__(
        self,
        name: str,
        *,
        owner: OwnerPath | None = None,
        state_name: str = "U",
        space: StateSpace | None = None,
    ) -> None:
        self.name = name
        self.owner_path = owner or OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, name)
        # A typed declaration with zero components is not an "unknown" StateSpace; it is a real
        # zero-width state and must correctly fail a scalar/vector solve.  Unit fixtures that do not
        # provide a model therefore use the smallest honest state, one named component.
        declared_space = space or StateSpace(state_name, (state_name,))
        self.state = StateHandle(
            state_name, owner=self.owner_path, space=declared_space)

    def declaration_index(self) -> DeclarationIndex:
        return DeclarationIndex(owner=self.owner_path, handles=(self.state,))


class _Context:
    def __init__(self, program: Program) -> None:
        self.program = program
        self.problem = Problem(name="typed-time-unit-case")
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


def _bound_owner_and_space(program: Program) -> tuple[Any, Any]:
    registries = getattr(program, "_operator_registries", {})
    if len(registries) != 1:
        return None, None
    owner = next(iter(registries))
    return owner, getattr(program, "_default_state_spaces", {}).get(owner)


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
            if selected_space is None:
                owner = selected_model.owner_path
                selected_space = getattr(program, "_default_state_spaces", {}).get(owner)
        model_key: Any = ("explicit", id(selected_model))
    else:
        owner, inferred_space = _bound_owner_and_space(program)
        selected_space = space if space is not None else inferred_space
        model_key = ("proxy", owner, state_name, _space_key(selected_space))
        selected_model = context.models.get(model_key)
        if selected_model is None:
            selected_model = _StateModel(
                owner.name if owner is not None else "typed_time_model_%s" % state_name,
                owner=owner,
                state_name=state_name,
                space=selected_space,
            )
            context.models[model_key] = selected_model
        selected_state = selected_model.state

    block_key = (block_name, state_name, model_key)
    existing = context.blocks.get(block_key)
    if existing is not None:
        return existing
    block = context.problem.add_block(block_name, selected_model)
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
    temporal = program.state(block, declaration)
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


def typed_field(program: Program, name: str) -> Any:
    """Return a case-owned FieldHandle usable by ``Program.solve_fields``."""
    if not isinstance(name, str) or not name:
        raise TypeError("typed field fixture name must be a non-empty string")
    context = _context(program)
    existing = context.fields.get(name)
    if existing is not None:
        return existing
    from pops.fields import FieldProblem

    field = context.problem.add_field(FieldProblem(name=name))
    context.fields[name] = field
    return field


def commits_by_block(program: Program) -> dict[str, Any]:
    """Readable test projection of typed commit keys onto their block labels."""
    return {
        state.block_ref.local_id: value
        for state, value in program.commits().items()
    }


__all__ = [
    "commits_by_block", "fresh_state_refs", "state_refs", "typed_field", "typed_state",
]
