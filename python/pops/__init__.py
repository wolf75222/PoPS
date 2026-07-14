"""PoPS operator-first Python interface.

Authoring, validation, resolution and inspection are pure Python. Native code is authenticated and
loaded only when compilation, binding or execution needs it. The public lifecycle is deliberately
small and explicit: ``Model / Program / Case -> validate -> resolve -> compile -> bind -> run``.
"""
from __future__ import annotations

from ._api import bind, compile, resolve, run, validate
from ._inspect import explain, inspect
from ._version import __version__
from .physics.board import Model
from .problem import Case
from .runtime.run_report import RunReport, RunStopReason
from .time._program.api import Program


__all__ = [
    "Model",
    "Program",
    "Case",
    "RunReport",
    "RunStopReason",
    "validate",
    "inspect",
    "explain",
    "resolve",
    "compile",
    "bind",
    "run",
    "__version__",
]
