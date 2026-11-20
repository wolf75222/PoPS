"""The rate of a total on a moving configuration (G03 lot M1b, ROM-21; note §7.2, §7.6).

``moving_total_rate(program, name, state, members, member_rate, rate=)`` returns the rate of the
cell totals U of one block on the moving configuration whose members (x_left, x_right, V) are the
state of another block (G03 lot M1a). It reads three values of one stage:

- ``state``: the totals U of the stage;
- ``members``: the members of the same stage, whose volume V decodes the density ū = U/V (that
  their block holds a moving configuration is checked at resolution);
- ``member_rate``: the rate of those members at that stage (``MovingMembers.rate(members)``), whose
  node components give the rate w of each face: the copy x_left of a cell for its lower face, the
  copy x_right for its upper face. The two copies of a node are bitwise equal, so the two cells of
  a face read the same w.

The rate of U_K is -(F_{K+1/2} - F_{K-1/2}), undivided, with F the Rusanov flux of F(ū) - w ū at
the speed |λ - w|, F and λ the flux and wave speed of ``rate``, a FiniteVolume rate of the model of
the block. Reading w from the members' own rate makes the swept volume of the faces of K, over a
stage, the rate of V_K bit for bit (CF11).
"""
from __future__ import annotations

from typing import Any

OPERATION = "moving_total_rate"


def _same_stage(left: Any, right: Any) -> bool:
    from pops.time._evaluation_point import evaluation_time_point
    from pops.time.points import StagePoint

    if type(left.point) is StagePoint and type(right.point) is StagePoint and left.point != right.point:
        return False
    try:
        a, b = evaluation_time_point(left), evaluation_time_point(right)
    except ValueError:
        return False
    return (a.clock, a.step, a.offset.to_python()) == (b.clock, b.step, b.offset.to_python())


def member_source(value: Any) -> Any:
    """Follow only a coefficient-one materialization at the exact evaluation point."""
    while value.op == "linear_combine" and len(value.inputs) == 1:
        (source,) = value.inputs
        coeffs = value.attrs.get("coeffs", ())
        if (len(coeffs) != 1 or dict(coeffs[0]) != {0: 1} or value.block != source.block
                or not _same_stage(value, source)):
            break
        value = source
    return value


def moving_total_rate(program: Any, name: Any, state: Any, members: Any, member_rate: Any, *,
                      rate: Any, members_owner: Any = None) -> Any:
    from pops.time._program.value_validation import rate_space_for, require_owned

    require_owned(program, state, "moving total rate state")
    require_owned(program, members, "moving total rate members")
    require_owned(program, member_rate, "moving total rate of the members")
    from pops.time.references import canonical_handle

    validate_inputs(state, members, member_rate, rate)
    owner = canonical_handle(members.state_ref)
    if members_owner is not None and owner != canonical_handle(members_owner):
        raise ValueError("moving transport differs from the declared owner of the moving members")
    return program._new("rhs", OPERATION, (state, members, member_rate),
                        {"operator_handle": rate, "members_block": members.block, "members_state": owner},
                        name, state.block, space=rate_space_for(state.space), point=state.point,
                        field_context=state.field_context)


def validate_inputs(state: Any, members: Any, member_rate: Any, rate: Any) -> None:
    """The three inputs of one stage, the members of a moving configuration, and a rate of the
    model of the block of ``state``."""
    from pops.mesh.moving_configuration import MOTION_SOURCE
    from pops.model.handles import OperatorHandle

    for value, what in ((state, "the totals"), (members, "the members")):
        if value.vtype != "state" or value.state_ref is None:
            raise TypeError("a moving total rate reads %s as a qualified State" % what)
    if members.block == state.block:
        raise ValueError("a moving total rate reads the members of another block")
    if members.point != state.point:
        raise ValueError("a moving total rate reads the totals and the members of one stage")
    source = member_source(member_rate)
    if (source.op != "source" or source.block != members.block or len(source.inputs) != 1
            or not _same_stage(member_rate, members) or source.inputs[0] is not members
            or source.attrs.get("source") != MOTION_SOURCE):
        raise ValueError("a moving total rate reads the rate of its members at the same stage, "
                         "the value of MovingMembers.rate(members)")
    if not isinstance(rate, OperatorHandle) or rate.kind != "local_rate":
        raise TypeError("a moving total rate takes rate= the FiniteVolume rate of the model of "
                        "the block of its totals")
    block = getattr(state.state_ref, "block_ref", None)
    owner = getattr(block, "model_owner_path", None)
    if owner is not None and rate.owner_path != owner:
        raise ValueError("rate= %r is not a rate of the model of the block of the totals"
                         % rate.name)


def validate_moving_total_rate(value: Any) -> None:
    if value.op != OPERATION or len(value.inputs) != 3:
        raise ValueError("moving total rate inputs changed")
    from pops.time.references import canonical_handle

    state, members, member_rate = value.inputs
    if (value.block != state.block or value.attrs.get("members_block") != members.block
            or value.attrs.get("members_state") != canonical_handle(members.state_ref)):
        raise ValueError("moving total rate block authority changed")
    validate_inputs(state, members, member_rate, value.attrs.get("operator_handle"))


__all__ = ["OPERATION", "member_source", "moving_total_rate", "validate_moving_total_rate"]
