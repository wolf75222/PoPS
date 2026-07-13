"""PoPS operator-first Python interface.

Authoring, validation, resolution and inspection are pure Python. Native code is authenticated and
loaded only when compilation, binding or execution needs it. The public lifecycle is deliberately
small and explicit: ``Model / Program / Case -> validate -> resolve -> compile -> bind -> run``.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

from ._api import bind, compile, resolve, run, validate
from ._inspect import explain, inspect
from ._version import __version__
from .physics.board import Model
from .problem import Case
from .time.program import Program


__all__ = [
    "Model",
    "Program",
    "Case",
    "validate",
    "inspect",
    "explain",
    "resolve",
    "compile",
    "bind",
    "run",
    "__version__",
]


_SUBMODULES = frozenset({
    "amr", "boundary", "codegen", "diagnostics", "domain", "external", "fields", "frames",
    "identity", "initial", "interfaces", "ir", "layouts", "lib", "linalg", "math", "mesh",
    "model", "moments", "numerics", "output", "params", "physics", "projection",
    "representations", "restart", "runtime", "schedule", "solvers", "spaces", "time",
})

def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        try:
            module = import_module("pops." + name)
        except ModuleNotFoundError as exc:
            if exc.name == "pops." + name:
                raise AttributeError("module 'pops' has no public submodule %r" % name) from None
            raise
        globals()[name] = module
        return module
    raise AttributeError("module 'pops' has no public attribute %r" % name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBMODULES)
