"""What the members of a moving configuration are served by (G03 lot M1a, ROM-21).

The members X and V of a moving configuration (``pops.mesh.moving_configuration``) are the state
of a block of their own, advanced by the step's own method. Two consumers of a case that holds them
are refused at resolution, before any C++, by name and with the lot that serves them:

- a step rule: the native step selection reads the base spacing and the speed of the stored state,
  while the bound that holds on a moving cell is on its current spacing and |λ - w·n|; a contracting
  cell would get a step that is too large. A moving configuration takes ``dt=`` a fixed step until
  lot T (note §7.9, ruling B2);
- a hierarchy of levels, and so its regrid: the transfer of a geometry under regrid is G11's
  (note §7.7, ruling C1).
"""
from __future__ import annotations

from typing import Any

from pops.time._program.traced_method import NotYetElaborated


class MovingMembersNotYetElaborated(NotYetElaborated):
    """A consumer of a moving configuration that lot M1 does not serve; ``reason`` names which."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def member_blocks(plan: Any) -> tuple[Any, ...]:
    """The blocks of a resolved plan whose state is the members of a moving configuration."""
    from pops.mesh.moving_configuration import members_of
    return tuple(block for block in plan.blocks if members_of(block.model) is not None)


def validate_moving_member_consumers(plan: Any) -> None:
    """Refuse a step rule and a hierarchy on a case that holds a moving configuration."""
    from pops.time._step.strategy import FixedDt

    blocks = member_blocks(plan)
    if not blocks:
        return
    name = blocks[0].name
    if any(target == "amr_system" for target in plan.layout_targets.values()):
        raise MovingMembersNotYetElaborated(
            "moving_regrid",
            "block %r holds a moving configuration; a hierarchy of levels and its regrid are not "
            "served on one: the transfer of a geometry under regrid is G11's" % name)
    strategy = None if plan.time is None else getattr(plan.time, "_step_strategy", None)
    if type(strategy) is not FixedDt:
        raise MovingMembersNotYetElaborated(
            "moving_step_rule",
            "block %r holds a moving configuration; the step strategy %s is not served on one, "
            "since the native step selection reads the base spacing and the speed of the stored "
            "state, not the current spacing and |lambda - w.n|; write dt= a fixed step until "
            "G03 lot T" % (name, type(strategy).__name__))


__all__ = ["MovingMembersNotYetElaborated", "member_blocks", "validate_moving_member_consumers"]
