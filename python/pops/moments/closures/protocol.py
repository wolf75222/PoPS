"""pops.moments.closures.protocol -- the moment-closure typing protocol.

A closure is the ONLY physics of a moment model: a callable that maps the standardized
moments ``S`` (a dict of ``S{p}{q}`` for ``2 <= p+q <= order``) to the standardized
moments of order ``order+1`` (a dict of ``S{p}{q}`` for ``p+q == order+1``). The values
may be DSL expressions or plain numbers (a numeric zero drops the term from the flux).
"""
import typing


@typing.runtime_checkable
class Closure(typing.Protocol):
    """A moment-closure callable: ``S -> dict`` of the order ``N+1`` standardized moments.

    ``__call__(S)`` receives the let-bound standardized moments ``S`` (keys ``S{p}{q}``,
    ``2 <= p+q <= order``, with ``S20 == S02 == 1``) and returns exactly the keys
    ``S{p}{q}`` for ``p+q == order+1``. The values are DSL expressions or numbers; a
    numeric zero removes the term from the generated flux.
    """

    def __call__(self, S):  # noqa: N803  (S mirrors the engine variable name)
        ...


__all__ = ["Closure"]
