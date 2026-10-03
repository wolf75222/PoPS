"""pops.moments.closures -- the moment-closure surface (the only physics).

Public surface:
  closure          -- a decorator that validates a closure returns exactly the order
                      ``N+1`` standardized-moment keys (mirrors the engine ``want`` check).
  Closure          -- the closure typing.Protocol.
  gaussian_closure -- the generic Gaussian / Levermore closure (provided).
  HyQMOM15Closure  -- the polynomial order-4 HyQMOM closure for Vlasov-Poisson.
  hyqmom_b1_axial_flux -- its explicitly selected order-4 one-velocity marginal.
"""
from __future__ import annotations

from typing import Any

from .protocol import Closure, LocalClosure, apply_local_closure
from .gaussian import gaussian_closure
from .discrete_entropy import DiscreteEntropyQuadrature
from .entropy_domain import DiscreteEntropyCertificate


def closure(order: Any) -> Any:
    """Decorate a moment closure, validating its returned keys against the order (criterion).

    A closure for a model of order ``order`` must return EXACTLY the keys ``S{p}{q}`` with
    ``p+q == order+1`` (the same ``want`` check :func:`build_moment_model` applies). This
    decorator wraps the callable so a wrong key set raises a clear :class:`TypeError` at the
    point the closure is evaluated, rather than the generic engine ``ValueError`` deeper in
    the build. The wrapped callable is otherwise unchanged -- it still returns the dict the
    engine consumes.

    @p order: the model order N (the closure supplies the order ``N+1`` standardized moments).
    """
    if not isinstance(order, int) or isinstance(order, bool) or order < 2:
        raise ValueError("@closure(order): order must be an int >= 2 (got %r)" % (order,))
    def decorate(fn: Any) -> Any:
        if not callable(fn):
            raise TypeError("@closure must decorate a callable closure; got %r" % (fn,))
        return LocalClosure(order, getattr(fn, "__name__", type(fn).__name__), fn)

    return decorate


# Import after the generic closure protocol is bound.
from .hyqmom15 import HyQMOM15Closure  # noqa: E402
from .hyqmom_axial_b1 import hyqmom_b1_axial_flux  # noqa: E402

__all__ = [
    "closure", "Closure", "LocalClosure", "apply_local_closure",
    "gaussian_closure", "HyQMOM15Closure", "hyqmom_b1_axial_flux",
    "DiscreteEntropyQuadrature", "DiscreteEntropyCertificate",
]
