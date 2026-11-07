"""Pre-implemented ``when`` bodies of ``Regrid(criterion, when)`` (spec 0.3, DM13b).

A ``when`` body is a closed body of the accepted continuation ``B`` (T33): it reads ``B.steps``,
``B.time`` and indicators of ``B.state`` and returns a Bool. It is traced once and generated into
the compiled Program; no Python runs in the step.
"""
from __future__ import annotations

from typing import Any, Callable


def every(k: int) -> Callable[[Any], Any]:
    """The ``when`` body true after every k-th accepted step: ``lambda B: B.steps % k == 0``."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("every(k) needs an integer k >= 1")

    def when(B: Any) -> Any:
        return B.steps % k == 0

    when.__name__ = "every_%d" % k
    when.__qualname__ = when.__name__
    return when


__all__ = ["every"]
