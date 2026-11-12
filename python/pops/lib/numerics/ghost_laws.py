"""The ghost laws of b6 that realize a boundary relation of the surface (N15 lot B, T5).

The elaborator of the surface checks, without naming them, that a boundary policy gives one of
the two ghost laws of master at each depth k: the mirror 2 g - u_k of a value g, and the copy u_1
of the first inner cell where the piece carries no data (2.6.9, D9). This module names the b6
condition of each, which the elaborator adds to the ``TransportBoundarySet`` of the block: it is
the bridge of the boundary policy, as ``SchemeValue.realize`` is the bridge of a scheme word, and
it goes away when G04 generates the ghost fills from the body of the policy.
"""
from __future__ import annotations

from typing import Any


def realize(form: str, state: Any, value: Any, spelled_state: str,
            spelled_value: str) -> tuple[Any, str, str]:
    """The b6 condition of a relation of form ``form`` on ``state`` with data ``value``, its
    spelling for ``pops.inspect`` and the ghost law it realizes."""
    from pops.boundary.transport import Inflow, Outflow

    if form == "Value":
        return (Inflow(state=state, value=value),
                "Inflow(state=%s, value=%s)" % (spelled_state, spelled_value),
                "2 g - u_k (FixedState)")
    if form == "NoData":
        return Outflow(state=state), "Outflow(state=%s)" % spelled_state, "u_1 (Extrapolate)"
    raise ValueError("no ghost law of b6 realizes the form %s" % form)


__all__ = ["realize"]
