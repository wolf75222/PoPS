"""Attach public pops.diagnostics state reductions to a ConsumerGraph.

Plan §2.3 / §7.0: accepted numerical state is reduced by public descriptors
embedded on a consumer. This helper does not reimplement those reductions.
"""
from __future__ import annotations

from typing import Any

from pops.diagnostics import (
    Balance,
    BalanceLedger,
    ConservationCheck,
    Integral,
    MinMax,
    Norm,
    StepChangeNorm,
)
from pops.linalg.norms import L1, L2, LInf
from pops.output import ConsoleMonitor, ConsumerGraph


def state_diagnostics(
    block: Any,
    cadence: Any,
    *,
    conservation_check: Any = None,
    ledger: Any = None,
) -> tuple[Any, ...]:
    """Return the public state-reduction descriptors for ``block`` at ``cadence``."""
    rows: list[Any] = [
        Integral(block=block, cadence=cadence),
        Norm(L1(), block=block, cadence=cadence),
        Norm(L2(), block=block, cadence=cadence),
        Norm(LInf(), block=block, cadence=cadence),
        MinMax(block=block, cadence=cadence),
        StepChangeNorm(L2(), block=block, cadence=cadence),
    ]
    if conservation_check is not None:
        if not isinstance(conservation_check, ConservationCheck):
            raise TypeError(
                "conservation_check must be a pops.diagnostics.ConservationCheck"
            )
        rows.append(conservation_check)
    if ledger is not None:
        if not isinstance(ledger, BalanceLedger):
            raise TypeError("ledger must be a pops.diagnostics.BalanceLedger")
        rows.append(Balance(ledger, block=block, cadence=cadence))
    return tuple(rows)


def attach_state_diagnostics(
    *,
    block: Any,
    cadence: Any,
    consumers: Any = (),
    conservation_check: Any = None,
    ledger: Any = None,
) -> ConsumerGraph:
    """Embed state diagnostics on a ConsoleMonitor and return a ConsumerGraph."""
    diagnostics = state_diagnostics(
        block,
        cadence,
        conservation_check=conservation_check,
        ledger=ledger,
    )
    monitor = ConsoleMonitor(schedule=cadence, diagnostics=diagnostics)
    return ConsumerGraph.from_consumers((monitor, *consumers))
