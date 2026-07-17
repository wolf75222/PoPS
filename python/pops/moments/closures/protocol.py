"""pops.moments.closures.protocol -- the moment-closure typing protocol.

A closure is the ONLY physics of a moment model: a callable that maps the standardized
moments ``S`` (a dict of ``S{p}{q}`` for ``2 <= p+q <= order``) to the standardized
moments of order ``order+1`` (a dict of ``S{p}{q}`` for ``p+q == order+1``). The values
may be DSL expressions or plain numbers (a numeric zero drops the term from the flux).
"""
from __future__ import annotations

import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@typing.runtime_checkable
class Closure(typing.Protocol):
    """A moment-closure callable: ``S -> dict`` of the order ``N+1`` standardized moments.

    ``__call__(S)`` receives the let-bound standardized moments ``S`` (keys ``S{p}{q}``,
    ``2 <= p+q <= order``, with ``S20 == S02 == 1``) and returns exactly the keys
    ``S{p}{q}`` for ``p+q == order+1``. The values are DSL expressions or numbers; a
    numeric zero removes the term from the generated flux.
    """

    def __call__(self, S: Any) -> Any:  # noqa: N803  (S mirrors the engine variable name)
        ...


def _closure_keys(order: int) -> frozenset[str]:
    return frozenset("S%d%d" % (p, order + 1 - p) for p in range(order + 2))


@dataclass(frozen=True, slots=True)
class LocalClosure:
    """A small generic local-algebra extension point.

    The evaluator is invoked once on symbolic standardized moments while the model is
    authored.  Its validated result is folded into the flux AST; the callable itself is
    never retained by the native hot path.  The contract contains no model-family token and
    therefore works for provided and user closures without a central dispatch table.
    """

    order: int
    name: str
    _evaluate: Callable[[Any], Any] = field(repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 2:
            raise ValueError("LocalClosure.order must be an int >= 2")
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("LocalClosure.name must be non-empty text")
        if not callable(self._evaluate):
            raise TypeError("LocalClosure evaluator must be callable")

    def __call__(self, standardized: Any) -> dict[str, Any]:
        result = self._evaluate(standardized)
        wanted = _closure_keys(self.order)
        if not isinstance(result, dict) or set(result) != wanted:
            got = sorted(result) if isinstance(result, dict) else type(result).__name__
            raise TypeError(
                "closure %r must return exactly the keys %s (got %s)"
                % (self.name, sorted(wanted), got)
            )
        return {key: result[key] for key in sorted(result)}

    def contract_data(self) -> dict[str, Any]:
        """Callback-free semantic data suitable for inspection and identities."""
        return {"kind": "local_moment_closure", "order": self.order, "name": self.name}


def apply_local_closure(closure: Any, order: int, standardized: Any) -> dict[str, Any]:
    """Apply any structural closure through the single generic validation boundary."""
    if not callable(closure):
        raise TypeError("moment closure must implement __call__(standardized_moments)")
    declared_order = getattr(closure, "order", None)
    if declared_order is not None and declared_order != order:
        raise ValueError(
            "moment closure declares order %r but the hierarchy has order %r"
            % (declared_order, order)
        )
    result = closure(standardized)
    wanted = _closure_keys(order)
    if not isinstance(result, dict) or set(result) != wanted:
        got = sorted(result) if isinstance(result, dict) else type(result).__name__
        raise TypeError(
            "moment closure must return exactly the keys %s (got %s)"
            % (sorted(wanted), got)
        )
    return {key: result[key] for key in sorted(result)}


# A custom moment closure is AUTHORING-ONLY. It is evaluated exactly once, at BUILD time,
# over the symbolic standardized moments ``S`` (DSL ``Expr`` inputs, not floats), and its
# output is folded into the flux AST that lowers to C++. There is NO Python on the production
# per-cell path: the closure's arithmetic becomes native primitives, exactly as a custom flux
# or a partial-DSL brick is authored symbolically and compiled. So a user closure upholds the
# bit-identity and no-Python-in-hot-paths rules without any runtime gate -- it either lowers to
# native (the common case) or fails at build; it can never reach the hot loop.


__all__ = ["Closure", "LocalClosure", "apply_local_closure"]
