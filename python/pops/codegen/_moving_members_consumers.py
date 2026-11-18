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

from collections.abc import Mapping
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


def moving_state_ids(blocks: Any) -> frozenset[str]:
    from pops.mesh.moving_configuration import members_of

    return frozenset(identity for block in blocks if members_of(block.model) is not None
                     for identity in block.state_identities)


def lower_moving_member_guards(program: Any, blocks: Any) -> Any:
    """Check physical geometry before evaluating its rate and before publishing the candidate."""
    from pops.mesh.moving_configuration import MEMBER_COMPONENTS
    from pops.time._program.control import AdvanceControl
    from pops.time._step.strategy import FixedDt
    from pops.time.references import canonical_handle

    state_ids = moving_state_ids(blocks)
    if not state_ids or type(program._step_strategy) is not FixedDt:
        return program
    volume_component = MEMBER_COMPONENTS.index("volume")
    trial_sites: list[dict[str, Any]] = []

    def is_geometry(state: Any) -> bool:
        return state is not None and canonical_handle(state).qualified_id in state_ids

    def guard(out: Any, value: Any, name: str) -> Any:
        volume = out._new("scalar", "reduce", (value,),
                          {"kind": "min", "comp": volume_component}, None, value.block)
        finite = out.isfinite(out.norm_inf_all(value))
        site = {"site": name, "variant": "NotAdmissible", "retry_can_change": False}
        result = out._trial_guard(name, value, finite & (volume > 0), site)
        trial_sites.append({**site, "guard": "admissibility", "classified": True, "lot": None})
        return result

    def guard_rate(out: Any, node: Any, inputs: list[Any]) -> list[Any]:
        # Equations evaluate their input as physical geometry. Method arithmetic can also
        # represent increments with the State role, so those edges carry no admissibility check.
        support_only = node.attrs.get("support_only_inputs", ())
        if node.op in {"source", "rhs", "pointwise_expression", "solve_implicit_source"}:
            return [guard(out, value, "moving geometry stage %d/%d" % (node.id, index))
                    if (index not in support_only and value.vtype == "state"
                        and is_geometry(value.state_ref)) else value
                    for index, value in enumerate(inputs)]
        return inputs

    def guard_commit(out: Any, state: Any, value: Any) -> Any:
        if is_geometry(state):
            return guard(out, value, "moving geometry commit %d" % value.id)
        return value

    guarded = program._rebuild(lambda _: True, inputs_of=guard_rate, commit_value=guard_commit)
    control = guarded._advance_control
    if control is not None:
        guarded._advance_control = AdvanceControl(
            assessments=control.assessments, max_attempts=control.max_attempts,
            max_transforms=control.max_transforms, first_dt=control.first_dt,
            outputs=control.outputs, memory=control.memory, step_rule=control.step_rule,
            trial_failures=(*control.trial_failures, *trial_sites),
        )
    guarded.validate()
    guarded.freeze()
    return guarded


def moving_geometry_bind_failures(
    blocks: Any, initial_values: Mapping[Any, Any], execution_context: Any,
) -> list[str]:
    """Report invalid member arrays before native installation, collectively under MPI."""
    import numpy as np

    component_by_id = {
        identity: block.moving_volume_component
        for block in blocks if block.moving_volume_component is not None
        for identity in block.state_identities
    }
    if not component_by_id:
        return []
    rejected = []
    for handle, array in initial_values.items():
        identity = handle.qualified_id
        component = component_by_id.get(identity)
        if (component is not None and isinstance(array, np.ndarray) and array.dtype.kind == "f"
                and array.ndim >= 2 and array.shape[0] > component
                and (not np.isfinite(array).all() or not (array[component] > 0).all())):
            rejected.append(identity)
    if execution_context is not None and execution_context.communicator.identity != "serial":
        from pops._native_collectives import allgather_value

        rows = allgather_value(execution_context.communicator.handle, tuple(rejected))
        rejected = [identity for row in rows for identity in row]
    return ["%s failed: NotAdmissible (geometry must be finite and cell volumes positive)"
            % identity for identity in sorted(set(rejected))]


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


__all__ = ["MovingMembersNotYetElaborated", "lower_moving_member_guards", "member_blocks",
           "moving_geometry_bind_failures", "validate_moving_member_consumers"]
