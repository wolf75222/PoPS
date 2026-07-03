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


#: The issue vocabulary spells the closure protocol ``MomentClosure``; it is the SAME
#: ``runtime_checkable`` Protocol (an identity alias, so ``isinstance`` checks are unchanged).
MomentClosure = Closure


# A custom moment closure is AUTHORING-ONLY. It is evaluated exactly once, at BUILD time,
# over the symbolic standardized moments ``S`` (DSL ``Expr`` inputs, not floats), and its
# output is folded into the flux AST that lowers to C++. There is NO Python on the production
# per-cell path: the closure's arithmetic becomes native primitives, exactly as a custom flux
# or a partial-DSL brick is authored symbolically and compiled. So a user closure upholds the
# bit-identity and no-Python-in-hot-paths rules without any runtime gate -- it either lowers to
# native (the common case) or fails at build; it can never reach the hot loop.


__all__ = ["Closure", "MomentClosure"]
