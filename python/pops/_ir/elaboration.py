"""One elaboration stack for every named body (spec 0.3 IR6; G02 lot L, design note §5.3).

A named body elaborates once: a GraphFunction while it is decorated, a ``@pops.solver`` while it
is lowered, a ``@pops.scheme`` while it is traced. Each enters :func:`elaborating` with its
definition. Entering a definition that is already on the stack raises
:class:`~pops._ir.refusals.RecursiveGraphFunction`, naming the cycle and the Python origin of each
call, instead of a Python ``RecursionError``. A Python helper that calls itself while it builds a
body is not a named definition: it only unrolls a finite body and enters nothing here.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
_CONTEXTLIB = os.path.abspath(contextlib.__file__)
_STACK: contextvars.ContextVar[tuple] = contextvars.ContextVar("pops_elaboration_stack", default=())


def _origin() -> str:
    """The first frame outside the package: where the user's code reached this elaboration."""
    frame = sys._getframe(1)
    while frame is not None:
        filename = os.path.abspath(frame.f_code.co_filename)
        if not filename.startswith(_PACKAGE) and filename != _CONTEXTLIB:
            return "%s:%d" % (filename, frame.f_lineno)
        frame = frame.f_back
    return "<unknown>"


@contextmanager
def elaborating(definition: Any, *, name: str) -> Iterator[None]:
    """Enter the elaboration of the named body @p definition (compared by identity)."""
    from .refusals import RecursiveGraphFunction

    stack = _STACK.get()
    entry = (definition, name, _origin())
    for position, (known, _, _) in enumerate(stack):
        if known is definition:
            raise RecursiveGraphFunction(
                tuple((label, origin) for _, label, origin in (*stack[position:], entry)))
    token = _STACK.set((*stack, entry))
    try:
        yield
    finally:
        _STACK.reset(token)
