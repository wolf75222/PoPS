"""Real typed state references for integration tests of the final Program API."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pops.model import Module
from pops.problem import Case


def module_of(model: Any) -> Module:
    module = getattr(model, "module", model)
    if not isinstance(module, Module):
        raise TypeError("integration fixture model must expose a pops.model.Module")
    return module


def state_handle(module: Module, state: Any = None) -> Any:
    spaces = module.state_spaces()
    if state is None:
        if len(spaces) != 1:
            raise ValueError("select a state explicitly for a multi-state Module")
        state = next(iter(spaces.values()))
    elif isinstance(state, str):
        state = spaces[state]
    return module.state_handle(state)


def synthetic_module(
    name: str,
    *,
    state_name: str = "U",
    components: tuple[str, ...] = (),
) -> Module:
    module = Module(name)
    module.state_space(state_name, components)
    return module


def program_states(
    program: Any,
    model: Any,
    block_names: Iterable[str],
    *,
    state: Any = None,
    case_name: str | None = None,
) -> tuple[Problem, dict[str, Any]]:
    """Declare every requested state through ``Program.state(BlockHandle, StateHandle)``."""
    module = module_of(model)
    declaration = state_handle(module, state)
    names = tuple(block_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("program_states requires non-empty block names")
    case = Case(name=case_name or "%s-program-case" % program.name)
    temporals = {
        name: program.state(case.block(name, module), declaration)
        for name in names
    }
    return case, temporals


__all__ = ["module_of", "program_states", "state_handle", "synthetic_module"]
