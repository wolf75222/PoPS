"""The :class:`RateBundle` typed multi-output of a coupled operator (Spec 2).

A coupled rate (``collisions(e, i, n) -> RateBundle``) returns one tangent per
participating block; :meth:`RateBundle.require` enforces that a block's rate lives
over the expected :class:`pops.model.spaces.StateSpace`.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any

from .spaces import Rate, RateSpace, Space


def _block_name(key: Any) -> Any:
    """The block/species name of a RateBundle key: a name string, or a space's name."""
    return key.name if isinstance(key, Space) else str(key)


class RateBundle:
    """A typed multi-output of a coupled operator: a mapping ``block -> Rate(StateSpace)``.

    A coupled rate (``collisions(e, i, n) -> RateBundle``) returns one tangent per
    participating block; ``bundle["electrons"]`` is the :class:`RateSpace` of that
    block. The arity is arbitrary (2, 3, 4, ... species). :meth:`require` enforces
    that a block's rate lives over the expected StateSpace. The full mapping is supplied at
    construction and then frozen, so a bundle remains a hash-stable Signature value.
    """

    __slots__ = ("_rates",)

    def __init__(self, entries: Any = None) -> None:
        rates = {}
        for block, rate in (entries or {}).items():
            name = _block_name(block)
            if name in rates:
                raise ValueError("RateBundle contains duplicate block %r" % (name,))
            rates[name] = rate if isinstance(rate, RateSpace) else Rate(rate)
        object.__setattr__(self, "_rates", MappingProxyType(rates))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("RateBundle is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("RateBundle is immutable")

    def require(self, block: Any, state: Any) -> Any:
        """Return the block's rate, raising if it is not ``Rate(state)`` (typed multi-output check)."""
        name = _block_name(block)
        got = self._rates.get(name)
        if got is None:
            known = ", ".join(self._rates) or "<none>"
            raise KeyError("RateBundle has no rate for block %r (have: %s)" % (name, known))
        want = Rate(state)
        if got != want:
            raise TypeError(
                "RateBundle[%r] is %r, not %r: a rate must live over its block's StateSpace"
                % (name, got, want))
        return got

    def __getitem__(self, block: Any) -> Any:
        return self._rates[_block_name(block)]

    def __contains__(self, block: Any) -> bool:
        return _block_name(block) in self._rates

    def keys(self) -> Any:
        return tuple(self._rates)

    def items(self) -> Any:
        return tuple(self._rates.items())

    def __len__(self) -> int:
        return len(self._rates)

    def _key(self) -> Any:
        # order-independent identity so a Signature output compares structurally
        return tuple(sorted(self._rates.items()))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, RateBundle) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return "RateBundle({%s})" % ", ".join(
            "%r: %r" % (k, v) for k, v in self._rates.items())

    def to_data(self) -> dict[str, Any]:
        return {"kind": "rate_bundle", "rates": [
            {"block": block, "rate": rate.to_data()} for block, rate in sorted(self._rates.items())
        ]}
