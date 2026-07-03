"""pops.moments.closures -- the moment-closure surface (the only physics).

Public surface:
  closure          -- a decorator that validates a closure returns exactly the order
                      ``N+1`` standardized-moment keys (mirrors the engine ``want`` check).
  Closure          -- the closure typing.Protocol.
  MomentClosure    -- the issue-vocabulary alias of ``Closure`` (the same Protocol).
  gaussian_closure -- the generic Gaussian / Levermore closure (provided).
  HyQMOM15Closure  -- the order-4 HyQMOM closure (Levermore variant) for vlasov_poisson.
"""
import functools

from .protocol import Closure, MomentClosure
from .gaussian import gaussian_closure


def closure(order):
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
    want = {"S%d%d" % (p, order + 1 - p) for p in range(order + 2)}

    def decorate(fn):
        if not callable(fn):
            raise TypeError("@closure must decorate a callable closure; got %r" % (fn,))

        @functools.wraps(fn)
        def wrapped(S):  # noqa: N803  (S mirrors the engine variable name)
            out = fn(S)
            if not isinstance(out, dict) or set(out) != want:
                raise TypeError(
                    "closure %r must return exactly the keys %s (got %s)"
                    % (getattr(fn, "__name__", fn), sorted(want),
                       sorted(out) if isinstance(out, dict) else type(out).__name__))
            return out

        wrapped.order = order
        return wrapped

    return decorate


# HyQMOM15Closure imports gaussian_closure, so import it after gaussian is bound.
from .hyqmom15 import HyQMOM15Closure  # noqa: E402

__all__ = ["closure", "Closure", "MomentClosure", "gaussian_closure", "HyQMOM15Closure"]
