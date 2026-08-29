"""Neutral nominal authority shared by Program lowering and checkpoint consumers.

The output layer must remain independent from code generation and the native runtime.  This
marker gives it a nominal, non-structural way to accept only a sealed lowering-owned resource
plan without importing either implementation layer.
"""

from __future__ import annotations


class ProgramResourcePlanCapacityAuthority:
    """Nominal base for the one sealed Program resource-plan implementation."""

    __slots__ = ()


__all__ = ["ProgramResourcePlanCapacityAuthority"]
