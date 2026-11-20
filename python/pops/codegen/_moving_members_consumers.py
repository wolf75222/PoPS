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


def lower_moving_member_guards(program: Any, blocks: Any) -> Any:
    """Check physical geometry before evaluating its rate and before publishing the candidate."""
    from contextlib import nullcontext

    from pops._ir.control_expr import Where
    from pops.mesh.moving_configuration import MEMBER_COMPONENTS, MOTION_SOURCE, members_of
    from pops.time._evaluation_point import evaluation_partition
    from pops.time._program.control import AdvanceControl
    from pops.time._program.moving_total_rate import OPERATION, member_source
    from pops.time._step.strategy import FixedDt
    from pops.time.references import canonical_handle

    geometries = {
        identity: members for block in blocks if (members := members_of(block.model)) is not None
        for identity in block.state_identities
    }
    if not geometries or type(program._step_strategy) is not FixedDt:
        return program
    volume_component = MEMBER_COMPONENTS.index("volume")
    left_component = MEMBER_COMPONENTS.index("x_left")
    right_component = MEMBER_COMPONENTS.index("x_right")
    incidence = moving_incidence_states(program)
    trial_sites: list[dict[str, Any]] = []
    guarded_rates: dict[int, tuple[int, Any]] = {}

    def geometry_of(state: Any) -> Any:
        return None if state is None else geometries.get(canonical_handle(state).qualified_id)

    def admit(out: Any, value: Any, condition: Any, name: str) -> Any:
        site = {"site": name, "variant": "NotAdmissible", "retry_can_change": False}
        result = out._trial_guard(name, value, condition, site)
        trial_sites.append({**site, "guard": "admissibility", "classified": True, "lot": None})
        return result

    def guard(out: Any, value: Any, name: str) -> Any:
        volume = out._new("scalar", "reduce", (value,),
                          {"kind": "min", "comp": volume_component}, None, value.block)
        finite = out.isfinite(out.norm_inf_all(value))
        protected = admit(out, value, finite & (volume > 0), name)
        if canonical_handle(value.state_ref).qualified_id not in incidence:
            return protected
        site = name + " orientation"
        status = out.value(
            site + " status",
            (Where(protected[left_component] < protected[right_component], 1, 0), 0, 0),
            at=protected.point,
        )
        return admit(out, protected, out.min(status) > 0, site)

    def guard_rate_domain(out: Any, node: Any, value: Any, members: Any) -> Any:
        reads_motion = ((node.op == "source" and node.attrs["source"] == MOTION_SOURCE)
                        or (node.op == "rhs" and (
                            node.attrs.get("sources") is None
                            or MOTION_SOURCE in node.attrs["sources"])))
        if not reads_motion or members.rate_domain is None:
            return value
        partition = node.attrs.get("evaluation_partition")
        if partition is None and "explicit" in getattr(node.point, "partitions", ()):
            partition = "explicit"
        with (nullcontext() if partition is None else evaluation_partition(out, partition)):
            status = out.transform(value, transform=members.rate_domain)
        return admit(out, value, out.min(status) > 0, "moving rate domain stage %d" % node.id)

    def guard_rate(out: Any, node: Any, inputs: list[Any]) -> list[Any]:
        # Equations evaluate their input as physical geometry. Method arithmetic can also
        # represent increments with the State role, so those edges carry no admissibility check.
        if node.op == OPERATION:
            source = member_source(node.inputs[2])
            record = guarded_rates.get(source.id)
            if record is None or record[0] != node.inputs[1].id:
                raise ValueError("moving total rate lost its exact geometry at guard lowering")
            return [inputs[0], record[1], inputs[2]]
        support_only = node.attrs.get("support_only_inputs", ())
        if node.op in {"source", "rhs", "pointwise_expression", "solve_implicit_source"}:
            checked = []
            for index, value in enumerate(inputs):
                members = geometry_of(value.state_ref)
                if index not in support_only and value.vtype == "state" and members is not None:
                    value = guard(out, value, "moving geometry stage %d/%d" % (node.id, index))
                    if index == 0:
                        value = guard_rate_domain(out, node, value, members)
                        if node.op == "source" and node.attrs["source"] == MOTION_SOURCE:
                            guarded_rates[node.id] = (node.inputs[0].id, value)
                checked.append(value)
            return checked
        return inputs

    def guard_commit(out: Any, state: Any, value: Any) -> Any:
        if geometry_of(state) is not None:
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


def moving_incidence_states(program: Any) -> frozenset[str]:
    """Exact state owners whose coordinates serve the emitted moving transport."""
    from pops.time._program.moving_total_rate import OPERATION
    from pops.time._program.traversal import all_ops

    return frozenset(value.attrs["members_state"].qualified_id for value in all_ops(program)
                     if value.op == OPERATION) if program is not None else frozenset()


def moving_geometry_incidence_failure(
    array: Any, volume_component: int, *, bounds: tuple[float, float] | None = None,
) -> str | None:
    """Check cold geometry in its installed Real precision, including shared-node bits."""
    import numpy as np
    from pops.mesh.moving_configuration import MEMBER_COMPONENTS

    if (array.ndim != 2 or array.size == 0 or array.shape[0] != len(MEMBER_COMPONENTS)
            or array.dtype.kind != "f" or array.dtype.itemsize not in (4, 8)):
        return "moving geometry incidence requires a complete rank-one Real image"
    left, right = (array[MEMBER_COMPONENTS.index(name)] for name in ("x_left", "x_right"))
    volume = array[volume_component]
    bits = np.dtype(f"u{array.dtype.itemsize}")
    for start in range(0, volume.size, 131072):
        end = min(start + 131072, volume.size)
        lo, hi, cell_volume = left[start:end], right[start:end], volume[start:end]
        if (not np.isfinite(lo).all() or not np.isfinite(hi).all()
                or not np.isfinite(cell_volume).all() or not np.all(lo < hi)
                or not np.all(cell_volume > 0)):
            return "moving geometry orientation and positive finite volume required"
        incident = min(end, volume.size - 1)
        same_node = (right[start:incident].view(bits) ==
                     left[start + 1:incident + 1].view(bits))
        if not same_node.all():
            face = start + int(np.argmax(~same_node)) + 1
            return "moving geometry incidence at face %d has unequal node bits" % face
        if bounds is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                if not np.array_equal(hi - lo, cell_volume):
                    return "moving geometry initial volume differs from the cell width"
    if bounds is not None and not np.array_equal(
            np.asarray((left[0], right[-1])), np.asarray(bounds, dtype=array.dtype)):
        return "moving geometry initial endpoints differ from the installed domain"
    return None


def moving_geometry_bind_failures(
    plan: Any, initial_values: Mapping[Any, Any], execution_context: Any,
) -> list[str]:
    """Report invalid member arrays before native installation, collectively under MPI."""
    import numpy as np

    geometry_by_id = {
        identity: block
        for block in plan.blocks if block.moving_volume_component is not None
        for identity in block.state_identities
    }
    if not geometry_by_id:
        return []
    rejected = []
    for handle, array in initial_values.items():
        identity = handle.qualified_id
        owner = geometry_by_id.get(identity)
        if owner is None:
            continue
        component = owner.moving_volume_component
        if owner.moving_incidence:
            from pops.time.references import canonical_handle

            layout = plan.layout_plan.layout_for(canonical_handle(handle.block_ref))
            geometry = plan.layout_plan.normalized(layout).geometry
            problem = moving_geometry_incidence_failure(
                np.asarray(array), component, bounds=(geometry.lower[0], geometry.upper[0]))
            if problem is not None:
                rejected.append((identity, problem))
        elif (isinstance(array, np.ndarray) and array.dtype.kind == "f"
              and array.ndim >= 2 and array.shape[0] > component
              and (not np.isfinite(array).all() or not (array[component] > 0).all())):
            rejected.append((identity, "geometry must be finite and cell volumes positive"))
    if execution_context is not None and execution_context.communicator.identity != "serial":
        from pops._native_collectives import allgather_value

        rows = allgather_value(execution_context.communicator.handle, tuple(rejected))
        rejected = [(identity, problem) for row in rows for identity, problem in row]
    return ["%s failed: NotAdmissible (%s)" % (identity, problem)
            for identity, problem in sorted(set(rejected))]


def moving_geometry_values(plan: Any, values: Any) -> frozenset[int]:
    """Find operations confined to one geometry, including its compiler-generated predicates."""
    from pops.time._program.moving_total_rate import OPERATION
    from pops.time.references import canonical_handle

    geometry_blocks = {block.name for block in member_blocks(plan)}
    if not geometry_blocks:
        return frozenset()
    owners = {canonical_handle(row.subject).qualified_id
              for row in plan.layout_plan.assignments if row.subject_kind == "block"
              and row.subject.local_id in geometry_blocks}
    dependencies: dict[int, frozenset[str]] = {}
    users: dict[int, list[tuple[Any, int]]] = {}
    for value in values:
        for index, source in enumerate(value.inputs):
            users.setdefault(source.id, []).append((value, index))

    def sources(value: Any) -> frozenset[str]:
        if value.id not in dependencies:
            block = value.block
            local = set() if block is None else {canonical_handle(block).qualified_id}
            inherited = frozenset().union(*(sources(source) for source in value.inputs))
            dependencies[value.id] = inherited | local
        return dependencies[value.id]

    owned = {value.id for value in values if len(sources(value)) == 1 and sources(value) <= owners}
    return frozenset(vid for vid in owned if all(
        other.id in owned or (other.op == OPERATION and index in (1, 2))
        for other, index in users.get(vid, ())
    ))


def validate_moving_member_consumers(plan: Any) -> None:
    """Refuse a step rule and a hierarchy on a case that holds a moving configuration."""
    from pops.time._step.strategy import FixedDt

    from pops.codegen.program_lowerability import all_ops
    from pops.time._program.moving_total_rate import OPERATION

    blocks = member_blocks(plan)
    if not blocks:
        if plan.time is not None and any(value.op == OPERATION for value in all_ops(plan.time)):
            raise MovingMembersNotYetElaborated(
                "moving_members_owner", "moving total transport requires an installed configuration")
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

    _validate_balances(plan, {block.name: block for block in blocks})

def _validate_balances(plan: Any, members: dict[str, Any]) -> None:
    from pops._ir.effects import EFFECTS
    from pops.codegen._cell_total_consumers import total_blocks
    from pops.codegen.program_lowerability import all_ops
    from pops.time._program.moving_total_rate import OPERATION

    layout_by_block = {row.subject.local_id: row.layout for row in plan.layout_plan.assignments
                       if row.subject_kind == "block"}
    layouts = {layout_by_block[name] for name in members}
    moving = {name for name, layout in layout_by_block.items() if layout in layouts}
    totals = {block.name for block in total_blocks(plan)}
    for block in plan.blocks:
        if block.name not in moving or block.name in members or block.name in totals:
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
        effect = EFFECTS.get(value.op)
        if block in moving and effect is not None and (effect.static_face or effect.cell_total == "density-rate"):
            raise MovingMembersNotYetElaborated(
                "moving_static_face",
                "the rate %r of block %r is a static face pass (%s), which reads the spacing of "
                "the base, not the members of the moving configuration; its rate is "
                "MovingMembers.transport(rate)(U, M, k_M) (G03 lot M1b)"
                % (value.name, block, value.op))
        if value.op == OPERATION:
            _validate_face(plan, value, block, members)


def _validate_face(plan: Any, value: Any, block: str, members: dict[str, Any]) -> None:
    """Resolve the exact co-located members and the physical face operator."""
    from pops.mesh.moving_configuration import members_of
    from pops.numerics.spatial import FiniteVolume
    from pops.time.references import canonical_handle

    member_block = value.attrs.get("members_block")
    member_name = getattr(member_block, "local_id", None)
    owner = members.get(member_name) if isinstance(member_name, str) else None
    geometry = value.attrs.get("members_state")
    if owner is None or geometry is None or geometry.qualified_id not in owner.state_identities:
        raise MovingMembersNotYetElaborated(
            "moving_members_owner", "the moving total rate reads no installed owner of the moving members")
    (resolved,) = [item for item in plan.blocks if item.name == block]
    state = canonical_handle(value.inputs[0].state_ref)
    if state.qualified_id not in resolved.state_identities or not resolved.numerics.cell_total_states():
        raise MovingMembersNotYetElaborated(
            "moving_total_storage", "moving total transport requires the exact CellTotal state")
    layout = plan.layout_plan.layout_for(canonical_handle(value.block))
    geometry_layout = plan.layout_plan.layout_for(canonical_handle(member_block))
    declared = members_of(owner.model)
    if (layout != geometry_layout or declared is None or resolved.numerics.interfaces
            or plan.layout_plan.normalized(layout).geometry != declared.configuration.layout.normalized_geometry()):
        raise MovingMembersNotYetElaborated(
            "moving_layout", "the totals and moving members require their co-located layout without interfaces")
    _validate_incidence_route(plan, resolved, declared.configuration, geometry, state, layout)
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


def _validate_incidence_route(
    plan: Any, block: Any, configuration: Any, owner: Any, state: Any, layout: Any,
) -> None:
    geometry = plan.layout_plan.normalized(layout)
    if (configuration.form != "expression"
            or geometry.options.get("embedded_boundary") is not None):
        raise MovingMembersNotYetElaborated(
            "moving_incidence_provider",
            "moving transport needs one position-based node law on a conforming Uniform layout")
    native = geometry.native_spatial_layout
    if native is None or len(native.decomposition["boxes"]) != 1:
        raise MovingMembersNotYetElaborated(
            "moving_incidence_patches",
            "moving transport needs a certified incidence across patches")
    initial = plan.initial_condition_plan
    routes = tuple(binding.source.options.to_data().get("native_route")
                   for binding in (() if initial is None else initial.bindings)
                   if binding.subject.qualified_id == owner.qualified_id)
    if routes != ("bound_level_zero",):
        raise MovingMembersNotYetElaborated(
            "moving_incidence_initial",
            "moving transport requires a BindArray for the exact geometry state")
    if not _has_two_interior_ghost_copies(block.numerics.boundaries, state, layout, geometry):
        raise MovingMembersNotYetElaborated(
            "moving_incidence_closure",
            "moving transport requires a native copy of the interior state at both physical faces")


def _has_two_interior_ghost_copies(boundaries: Any, state: Any, layout: Any, geometry: Any) -> bool:
    from pops.boundary.transport import ResolvedTransportBoundarySet
    from pops.mesh.boundaries import ClosureMode, GhostProducerPlan, GhostState
    from pops.mesh.boundaries.topology import BoundarySide

    if len(boundaries) != 1 or type(boundaries[0]) is not GhostProducerPlan:
        return False
    plan = boundaries[0]
    authority = plan.execution_authority
    if (type(authority) is not ResolvedTransportBoundarySet
            or authority.frame_id != geometry.geometry.frame_id
            or len(authority.conditions) != 2 or len(authority.plan.bindings) != 2
            or plan.interfaces or plan.residual_contributions or plan.linearization_contributions
            or plan.component_bindings
            or plan.topology.boundaries != authority.plan.topology.boundaries
            or any(row.producer.periodic or row.producer.interfaces
                   or (row.region.boundary is None and row.producer.boundary_providers)
                   for row in plan.productions)):
        return False
    physical = tuple(row for row in plan.productions if row.region.boundary is not None)
    if len(physical) != 2:
        return False
    sides = set()
    for row in physical:
        region = row.region
        boundary = region.boundary
        if region.subject != state or region.layout != layout or boundary is None:
            return False
        orientation = boundary.orientation
        providers = row.producer.boundary_providers
        if orientation.axis != 0 or orientation.side in sides or len(providers) != 1:
            return False
        sides.add(orientation.side)
        (provider,) = providers
        conditions = tuple(condition for condition in authority.conditions
                           if condition.provider is provider)
        if len(conditions) != 1:
            return False
        (condition,) = conditions
        dependency = provider.dependencies
        if (not condition.copies_nearest_interior or condition.state != state
                or condition.geometry.domain_geometry_id != authority.domain_geometry_id
                or condition.geometry.axis.index != orientation.axis
                or condition.geometry.side.value != orientation.side.value
                or condition.requirement.ghost_depth < 1 or region.depth.depth[0] < 1
                or condition.prescribed_trace or condition.values or condition.source_ghost is not None
                or dependency.states != (state,) or dependency.fields or dependency.time
                or dependency.runtime_params or dependency.characteristic.mode is not ClosureMode.NONE
                or dependency.representation.source != dependency.representation.target
                or dependency.representation.converter is not None or len(provider.outputs) != 1):
            return False
        (port,) = provider.outputs
        if (type(port) is not GhostState or port.subject != state or port.boundary != boundary
                or not any(binding.provider is provider and binding.need == port
                           for binding in authority.plan.bindings)):
            return False
    return sides == {BoundarySide.LOWER, BoundarySide.UPPER}


__all__ = ["MovingMembersNotYetElaborated", "lower_moving_member_guards", "member_blocks",
           "moving_geometry_bind_failures", "validate_moving_member_consumers"]
