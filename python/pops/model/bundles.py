"""The :class:`RateBundle` typed multi-output of a coupled operator (Spec 2).

A coupled rate (``collisions(e, i, n) -> RateBundle``) returns one tangent per
participating block; :meth:`RateBundle.require` enforces that a block's rate lives
over the expected :class:`pops.model.spaces.StateSpace`.
"""
from __future__ import annotations

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
    that a block's rate lives over the expected StateSpace, so adding a
    ``Rate(electron_state)`` where a ``Rate(ion_state)`` is expected is rejected.
    """

    def __init__(self, entries: Any = None) -> None:
        self._rates = {}
        for block, rate in (entries or {}).items():
            self.add(block, rate)

    def add(self, block: Any, rate: Any) -> Any:
        """Bind ``block`` to ``rate`` (a :class:`RateSpace`, a :class:`StateSpace`, or a name)."""
        rs = rate if isinstance(rate, RateSpace) else Rate(rate)
        self._rates[_block_name(block)] = rs
        return self

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
        return list(self._rates)

    def items(self) -> Any:
        return list(self._rates.items())

    def __len__(self) -> int:
        return len(self._rates)

    def _key(self) -> Any:
        # order-independent identity so a Signature output compares structurally
        return tuple(sorted((k, repr(v)) for k, v in self._rates.items()))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, RateBundle) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return "RateBundle({%s})" % ", ".join(
            "%r: %r" % (k, v) for k, v in self._rates.items())
