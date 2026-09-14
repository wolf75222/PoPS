"""Prove and emit the accepted affine quadrature of joint physical inventory rows."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
import json


def _polynomial(value):
    from .program_emit_kernels import _coeff_metadata_terms
    return {power: Fraction(numerator, denominator)
            for power, numerator, denominator in _coeff_metadata_terms(value)}


def _multiply(left, right):
    result = defaultdict(Fraction)
    for p, a in left.items():
        for q, b in right.items():
            result[p + q] += a * b
    return {power: value for power, value in result.items() if value}


def accepted_interaction_quadrature(program):
    """Follow actual accepted SSA affine edges, never authoring-time intended weights.

    This direct realization supports affine explicit updates. A nonlinear transform,
    branch, history, or solved update needs its own proved quadrature propagation.
    Presence of an inventory claim cannot silently traverse such an operation.
    """
    records, memo = {}, {}

    def propagate(value):
        value = program._canonical_value(value)
        if value.id in memo:
            return memo[value.id]
        rows = value.attrs.get("interaction_occurrences")
        result = {}
        if rows:
            for row in rows:
                balance, ordinal = row["occurrence"]
                key = (row["evaluation_id"], balance, ordinal)
                records[key] = row
                result[key] = {0: Fraction(row["coefficient"])}
        elif value.op == "linear_combine":
            for source, coefficient in zip(value.inputs, value.attrs["coeffs"], strict=True):
                for key, weights in propagate(source).items():
                    destination = result.setdefault(key, defaultdict(Fraction))
                    for power, amount in _multiply(weights, _polynomial(coefficient)).items():
                        destination[power] += amount
        elif value.op in ("state", "rhs", "source", "coupled_rate", "coupled_rate_out") or (
                value.op == "apply" and value.vtype == "rhs"):
            result = {}
        elif value.op in ("acceptance_guard", "solve_outcome") and value.inputs:
            result = propagate(value.inputs[0])
        elif value.inputs:
            if any(propagate(source) for source in value.inputs):
                raise ValueError("accepted interaction inventory crosses an unproved %s quadrature" % value.op)
        memo[value.id] = {key: {power: weight for power, weight in weights.items() if weight}
                          for key, weights in result.items() if any(weights.values())}
        return memo[value.id]

    accepted = {}
    for endpoint, value in program._commits.items():
        for key, weights in propagate(value).items():
            row = records[key]
            if endpoint.declaration_ref != row["recipient"]:
                raise ValueError("accepted inventory occurrence reaches a foreign state endpoint")
            if key in accepted:
                raise ValueError("one interaction occurrence reaches multiple accepted endpoints")
            accepted[key] = weights
    groups = {}
    for key, weights in accepted.items():
        row = records[key]
        for inventory in row["inventories"]:
            group = groups.setdefault((key[0], inventory.name), {
                "inventory": inventory, "application": row["application"],
                "recipients": defaultdict(lambda: defaultdict(Fraction)), "rows": []})
            if group["inventory"].to_data() != inventory.to_data():
                raise ValueError("inventory name aliases different physical projections")
            if row["recipient"] not in {projection.state for projection in inventory.projections}:
                continue
            for power, amount in weights.items():
                group["recipients"][row["recipient"]][power] += amount
            group["rows"].append((row, weights))
    result = []
    for group in groups.values():
        inventory, recipients = group["inventory"], group["recipients"]
        states = {projection.state for projection in inventory.projections}
        if set(recipients) != states:
            raise ValueError("accepted interaction inventory is missing a committed recipient")
        powers = {power for weights in recipients.values() for power in weights}
        if powers - {1}:
            raise ValueError("accepted interaction inventory requires a proved weight * dt quadrature")
        for power in powers:
            inventory.require_accepted_quadrature(group["application"],
                {state: recipients[state].get(power, Fraction(0)) for state in states})
        result.extend((inventory, row, weights) for row, weights in group["rows"])
    return tuple(result)


def emit_accepted_interaction_exchanges(program, var, block_indices, *, target):
    rows = accepted_interaction_quadrature(program)
    if not rows:
        return []
    if target != "system":
        raise NotImplementedError("joint inventory physical-volume realization requires uncut Uniform Cartesian cells")
    from .program_emit_kernels import _coeff_cpp
    from pops.time.references import canonical_handle
    from pops.identity.scalar import scalar_cpp
    lines = ["pops::Real joint_cell_volume = 1;",
             "for (int axis = 0; axis < pops::kNativeDimension; ++axis) "
             "joint_cell_volume *= ctx.geometry().spacing(axis);"]
    for ordinal, (inventory, row, weights) in enumerate(rows):
        evaluation = next(value for value in program._values if value.id == row["evaluation_id"])
        recipient = row["recipient"]
        projection, = [p for p in inventory.projections if p.state == recipient]
        block = evaluation.attrs["output_bindings"][recipient.local_id]
        block_index = block_indices[block]
        scratch = var[("coupled_scratch", evaluation.id, block)]
        terms = ["(%s) * ctx.sum_component(%d, %s, %d)" %
                 (scalar_cpp(weight), block_index, scratch, component)
                 for component, weight in enumerate(projection.weights) if weight]
        amount = " + ".join(terms) or "pops::Real(0)"
        balance, occurrence = row["occurrence"]
        operation = canonical_handle(row["application"].operator).qualified_id
        context = "%s:evaluation:%d" % (operation, evaluation.id)
        quadrature = hashlib.sha256(json.dumps({
            "inventory": inventory.resolve_references(canonical_handle).to_data(),
            "weights": sorted((p, str(c)) for p, c in weights.items())
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        signed_weight = weights.get(1, Fraction(0))
        orientation = -1 if signed_weight < 0 else 1
        positive_weights = {1: abs(signed_weight)}
        lines.extend([
            "ctx.require_cartesian_generated_operator(%d, \"physical inventory exchange\");" % block_index,
            "const pops::Real joint_inventory_%d = %s;" % (ordinal, amount),
            "ctx.stage_exchange(pops::runtime::program::ExchangeRecord{%s, %s, %s, %s, "
            "%d, joint_cell_volume, joint_inventory_%d, %s, 1});" % (
                json.dumps(operation), json.dumps("%s:%d:%s" % (canonical_handle(balance).qualified_id, occurrence, inventory.name)),
                json.dumps(context), json.dumps(quadrature), orientation, ordinal, _coeff_cpp(positive_weights)),
        ])
    return lines
