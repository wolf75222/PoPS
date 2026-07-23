"""Typed diagnostic declarations.

Historical lowercase descriptor factories are intentionally absent: diagnostics are authored
with immutable typed measures and attached to the Case consumer graph.
"""
from .invariants import invariants
from .measures import ConservationCheck, Integral, MinMax, Norm, StepChangeNorm

__all__ = [
    "ConservationCheck", "Integral", "MinMax", "Norm", "StepChangeNorm", "invariants",
]
