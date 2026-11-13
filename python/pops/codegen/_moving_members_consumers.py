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

The other blocks of such a case are balances on the moving cells (G03 lot M1b):

- each holds its state as cell totals (``cell_total(1)``): an average on a moving cell is not
  served, the case writes ``represent={U: pops.CellTotal()}``, or an accumulation of lot A;
- its rate is ``moving_total_rate`` (``MovingMembers.transport``); a static face pass (a native rate,
  a diffusion, a stencil flux, a principal or coupled rate) reads the spacing of the base, not the
  members, and is refused;
- the face of ``moving_total_rate`` is first order and Rusanov, of a FiniteVolume rate.
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


# The rates whose face pass reads the static spacing of the base.
def _static_face_ops() -> frozenset:
    from pops.time._program.stencil_flux import OPERATION as STENCIL
    return frozenset({"rhs", "diffusive_rhs", STENCIL, "principal_rate", "coupled_rate"})


STATIC_FACE_OPS = _static_face_ops()


def validate_moving_member_consumers(plan: Any) -> None:
    """Refuse a step rule and a hierarchy on a case that holds a moving configuration, and every
    balance on it that is not a total transported by ``moving_total_rate``."""
    from pops.time._step.strategy import FixedDt

    blocks = member_blocks(plan)
    if not blocks:
        return
    name = blocks[0].name
    _validate_balances(plan, {block.name for block in blocks})
    if any(target == "amr_system" for target in plan.layout_targets.values()):
        raise MovingMembersNotYetElaborated(
            "moving_regrid",
            "block %r holds a moving configuration; a hierarchy of levels and its regrid are not "
            "served on one: the transfer of a geometry under regrid is G11's" % name)
    if plan.time is None:
        raise MovingMembersNotYetElaborated(
            "moving_step_rule",
            "block %r holds a moving configuration, and the case has no Program; a moving "
            "configuration needs a Program with dt= a fixed step until G03 lot T" % name)
    strategy = getattr(plan.time, "_step_strategy", None)
    if type(strategy) is not FixedDt:
        raise MovingMembersNotYetElaborated(
            "moving_step_rule",
            "block %r holds a moving configuration; the step strategy %s is not served on one, "
            "since the native step selection reads the base spacing and the speed of the stored "
            "state, not the current spacing and |lambda - w.n|; write dt= a fixed step until "
            "G03 lot T" % (name, type(strategy).__name__))


def _validate_balances(plan: Any, members: set) -> None:
    from pops.codegen._cell_total_consumers import total_blocks
    from pops.codegen.program_lowerability import all_ops
    from pops.time._program.moving_total_rate import OPERATION

    totals = {block.name for block in total_blocks(plan)}
    for block in plan.blocks:
        if block.name in members or block.name in totals:
            continue
        states = ", ".join(str(getattr(state, "name", state))
                           for state in getattr(block, "state_spaces", ()))
        raise MovingMembersNotYetElaborated(
            "moving_balance_not_total",
            "block %r holds %s as cell averages on a moving configuration; a balance on moving "
            "cells holds its totals: write represent={U: pops.CellTotal()} for its state U (or an "
            "accumulation of G03 lot A)" % (block.name, states or "its state"))
    for value in () if plan.time is None else all_ops(plan.time):
        block = getattr(value.block, "local_id", None)
        if block is None or block in members:
            continue
        if value.op in STATIC_FACE_OPS:
            raise MovingMembersNotYetElaborated(
                "moving_static_face",
                "the rate %r of block %r is a static face pass (%s), which reads the spacing of "
                "the base, not the members of the moving configuration; its rate is "
                "MovingMembers.transport(rate)(U, M, k_M) (G03 lot M1b)"
                % (value.name, block, value.op))
        if value.op == OPERATION:
            _validate_face(plan, value, block, members)


def _validate_face(plan: Any, value: Any, block: str, members: set) -> None:
    """The members of a moving total rate, and its face: FiniteVolume, first order, Rusanov."""
    from pops.numerics.spatial import FiniteVolume

    if getattr(value.attrs.get("members_block"), "local_id", None) not in members:
        raise ValueError("the moving total rate %r reads a block that holds no moving "
                         "configuration" % value.name)
    (resolved,) = [item for item in plan.blocks if item.name == block]
    handle = value.attrs["operator_handle"]
    rows = [row for row in resolved.numerics.rates
            if row.rate.registered_operator_name == handle.registered_operator_name]
    method = rows[0].method if len(rows) == 1 else None
    if type(method) is not FiniteVolume:
        raise MovingMembersNotYetElaborated(
            "moving_face", "the moving total rate %r takes rate= a FiniteVolume rate of block %r"
            % (value.name, block))
    from pops.codegen.program_emit_moving_total import moving_total_entry
    from pops.codegen.program_models import ProgramModelGraph
    try:
        moving_total_entry(value, ProgramModelGraph.from_resolved_blocks(plan.blocks))
    except NotImplementedError as error:
        raise MovingMembersNotYetElaborated(
            "moving_face",
            "the moving total rate %r is served with FirstOrder and Rusanov on one state of rank "
            "one (%s); other faces on moving cells read the members in their stencil and their "
            "speeds, a later G03 lot" % (value.name, error)) from error


__all__ = ["MovingMembersNotYetElaborated", "STATIC_FACE_OPS", "member_blocks",
           "validate_moving_member_consumers"]
