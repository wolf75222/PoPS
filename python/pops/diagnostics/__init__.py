"""Typed diagnostic declarations.

Historical lowercase descriptor factories are intentionally absent: diagnostics are authored
with immutable typed measures and attached to the Case consumer graph.
"""
from .balance import BalanceLedger
from .invariants import invariants
from .measures import Balance, ConservationCheck, Integral, MinMax, Norm, StepChangeNorm

__all__ = [
    "Balance", "BalanceLedger", "ConservationCheck", "Integral", "MinMax", "Norm",
    "StepChangeNorm", "invariants",
]
