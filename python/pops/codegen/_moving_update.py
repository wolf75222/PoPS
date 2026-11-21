"""Cold proof of a common time update for totals and moving members (D21)."""
from __future__ import annotations

from typing import Any, NoReturn

from pops.identity.scalar import scalar_data
from pops.time._program.moving_total_rate import OPERATION, _same_stage, member_source
from pops.time._program.traversal import all_ops
from pops.time.references import canonical_handle

from .program_emit_schedule import _guarded_schedule_lowering


def _always(value: Any) -> bool:
    return value.region == 0 and _guarded_schedule_lowering(value) is None


def _ordered_coefficients(value: Any) -> list[Any]:
    return [[(power, scalar_data(number)) for power, number in sorted(term.items())]
            for term in value.attrs["coeffs"]]


def validate_moving_updates(program: Any) -> None:
    """Prove the actual ordered recurrence and joint publication of each transport."""
    from ._moving_members_consumers import MovingMembersNotYetElaborated

    if program is None:
        return
    transports = tuple(node for node in all_ops(program) if node.op == OPERATION)
    if not transports:
        return

    def refuse(where: str, cause: str) -> NoReturn:
        raise MovingMembersNotYetElaborated("moving_update", "moving %s: %s" % (where, cause))

    if (program._cadence is not None or program._cell_local_time is not None
            or program._post_sync_commits):
        refuse("update", "the totals and geometry need one explicit publication phase")

    commits = {canonical_handle(state): value for state, value in program._commits.items()}
    accepted = {value.id for value in program._time_current_values.values()}
    checked: set[tuple[int, int]] = set()

    def compare(total: Any, geometry: Any, where: str) -> None:
        pending = [(total, geometry)]
        while pending:
            left, right = pending.pop()
            pair = (left.id, right.id)
            if pair in checked:
                continue
            if not (_always(left) and _always(right) and _same_stage(left, right)):
                refuse(where, "stage or due operations differ at %s / %s" % (left.name, right.name))
            checked.add(pair)

            if (left.block != total.block or right.block != geometry.block):
                refuse(where, "different update owners at %s / %s" % (left.name, right.name))

            rate = member_source(left)
            if rate.op == OPERATION:
                if right.id != rate.inputs[2].id:
                    refuse(where, "different geometry rate at %s / %s" % (left.name, right.name))
                # Identical arithmetic is required even for a coefficient-one
                # combine: the ordinary emitter evaluates its AXPY.
                while left.id != rate.id:
                    if (right.op != "linear_combine" or len(right.inputs) != 1
                            or _ordered_coefficients(left) != _ordered_coefficients(right)
                            or not (_always(left) and _always(right) and _same_stage(left, right))):
                        refuse(where, "different rate materialization at %s / %s" %
                               (left.name, right.name))
                    left, right = left.inputs[0], right.inputs[0]
                if (right.id != member_source(rate.inputs[2]).id or right.op != "source"
                        or not (_always(left) and _always(right) and _same_stage(left, right))):
                    refuse(where, "different rate provenance at %s / %s" % (left.name, right.name))
                continue

            if left.op == right.op == "state" and left.id in accepted and right.id in accepted:
                if (canonical_handle(left.state_ref) != canonical_handle(total.state_ref)
                        or canonical_handle(right.state_ref) != canonical_handle(geometry.state_ref)):
                    refuse(where, "different accepted-state owners")
                continue

            if (left.op != "linear_combine" or right.op != "linear_combine"
                    or left.attrs.get("support_only_inputs") or right.attrs.get("support_only_inputs")):
                refuse(where, "no common explicit affine update at %s / %s" % (left.name, right.name))
            if _ordered_coefficients(left) != _ordered_coefficients(right):
                refuse(where, "different ordered coefficients at %s / %s" % (left.name, right.name))
            pending.extend(reversed(tuple(zip(left.inputs, right.inputs, strict=True))))

    for rate in transports:
        total, geometry, _ = rate.inputs
        if not _always(rate):
            refuse("stage %s" % rate.name, "transport is not an unconditional operation")
        compare(total, geometry, "stage %s / %s" % (total.name, geometry.name))

    owners = {(canonical_handle(rate.inputs[0].state_ref), canonical_handle(rate.inputs[1].state_ref))
              for rate in transports}
    for total_owner, geometry_owner in sorted(owners, key=lambda pair: tuple(h.qualified_id for h in pair)):
        total_commit = commits.get(total_owner)
        geometry_commit = commits.get(geometry_owner)
        if (total_commit is None or geometry_commit is None
                or total_commit.vtype != "state" or geometry_commit.vtype != "state"):
            refuse("commit", "both %s and %s need commits in the same candidate" %
                   (total_owner.block_ref.local_id, geometry_owner.block_ref.local_id))
        compare(total_commit, geometry_commit,
                "commit %s / %s" % (total_commit.name, geometry_commit.name))
        pending = [total_commit]
        visited: set[int] = set()
        while pending:
            node = pending.pop()
            if node.id in visited:
                continue
            visited.add(node.id)
            if node.op == OPERATION:
                break
            if node.op == "linear_combine":
                pending.extend(value for value, polynomial in
                               zip(node.inputs, node.attrs["coeffs"], strict=True)
                               if any(coefficient != 0 for coefficient in polynomial.values()))
        else:
            refuse("commit %s" % total_commit.name, "no effective moving rate contributes")
