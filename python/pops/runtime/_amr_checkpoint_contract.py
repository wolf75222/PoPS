"""Canonical accepted-state contract for strict AMR checkpoint/restart."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import struct

from pops.identity import make_identity


_SCHEMA = 7
_GUARANTEE = "bit_identical_accepted_state"
_CONTRACT_KEYS = {
    "schema_version",
    "guarantee",
    "program_state",
    "ledger",
    "interface_ledger",
    "clocks",
    "temporal_partition",
    "synchronization",
    "history_qualifications",
    "level_relations",
    "transfer_routes",
    "field_providers",
}
_PREFLIGHT_KEYS = {
    "schema_version",
    "guarantee",
    "program_state",
    "level_relations",
    "transfer_routes",
    "field_providers",
}


def _rows(values):
    return [list(map(str, row)) for row in values]


def _field_provider_contract_rows(values):
    """Retain immutable provider semantics, excluding live topology/materialization evidence."""
    rows = _rows(values)
    if any(len(row) < 14 or row[0] != "pops.amr.field-provider-checkpoint-manifest@1" for row in rows):
        raise ValueError("native AMR field-provider manifest has an invalid row")
    return [row[:6] + row[9:] for row in rows]


def _valid_field_recompute_dt(dt, accepted_step):
    """Accept the recorded dt, including only the honest initial accepted-point zero."""
    return math.isfinite(dt) and dt >= 0.0 and (dt > 0.0 or accepted_step == 0)


def restart_topology_image(sim):
    """Return the compact identity of one accepted AMR hierarchy."""
    levels = int(sim.n_levels())
    boxes = []
    for position, row in enumerate(sim.patch_boxes()):
        if type(row) is not tuple or len(row) != 3:
            raise TypeError(
                "native AMR patch_boxes row %d must be an exact (level, lower, upper) tuple"
                % position
            )
        level, lower, upper = row
        if type(level) is not int:
            raise TypeError("native AMR patch-box level %d must be an exact integer" % position)
        if type(lower) is not tuple or type(upper) is not tuple:
            raise TypeError("native AMR patch-box bounds must be exact ranked tuples")
        if len(lower) != len(upper) or not lower:
            raise ValueError("native AMR patch-box bounds have mismatched rank")
        if any(type(value) is not int for value in lower + upper):
            raise TypeError("native AMR patch-box bounds must contain exact integers")
        boxes.append([level, list(lower), list(upper)])
    owners = [[int(rank) for rank in sim.level_owner_ranks(level)] for level in range(levels)]
    topology_identity = make_identity(
        "restart-topology",
        {
            "active_levels": levels,
            "patch_boxes": boxes,
            "level_owner_ranks": owners,
        },
    ).token
    return {
        "topology_epoch": int(sim.checkpoint_topology_epoch()),
        "regrid_count": int(sim.checkpoint_regrid_count()),
        "active_levels": levels,
        "topology_identity": topology_identity,
    }


def contract_for(sim):
    """Return the audit-readable part of the native accepted-state image."""
    relation_rows = [list(map(str, row)) for row in sim.checkpoint_temporal_relations()]
    relations = []
    for row in relation_rows:
        if len(row) != 5:
            raise ValueError("native AMR temporal relation report has an invalid row")
        parent, child, numerator, denominator, remainder = row
        relations.append(
            {
                "parent": int(parent),
                "child": int(child),
                "temporal_ratio": {
                    "numerator": int(numerator),
                    "denominator": int(denominator),
                },
                "remainder_policy": remainder,
            }
        )
    flux_ledger = _rows(sim.program_flux_ledger_manifest())
    interface_flux_ledger = _rows(sim.program_interface_flux_ledger_manifest())
    return {
        "schema_version": _SCHEMA,
        "guarantee": _GUARANTEE,
        "program_state": "compiled" if sim.installed_program_hash() else "native_none",
        "ledger": {
            "accepted_entries": len(flux_ledger),
            "transaction_depth": 0,
            "entries": flux_ledger,
        },
        "interface_ledger": {
            "accepted_entries": len(interface_flux_ledger),
            "transaction_depth": 0,
            "entries": interface_flux_ledger,
        },
        "clocks": _rows(sim.program_clock_manifest()),
        "temporal_partition": _rows(sim.program_temporal_partition_manifest()),
        "synchronization": _rows(sim.program_sync_manifest()),
        "history_qualifications": _rows(sim.program_accepted_state_manifest()),
        "level_relations": relations,
        "transfer_routes": _rows(sim.checkpoint_transfer_routes()),
        "field_providers": _field_provider_contract_rows(
            sim.field_provider_checkpoint_manifest()
        ),
    }


def encode_contract(sim):
    return json.dumps(contract_for(sim), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decode_contract(payload):
    from pops._manifest_protocol import strict_json_loads

    contract = strict_json_loads(
        str(payload["amr_accepted_contract"]), where="AMR accepted-state contract"
    )
    if not isinstance(contract, dict) or set(contract) != _CONTRACT_KEYS:
        raise TypeError("restart: AMR accepted-state contract has an invalid exact schema")
    return contract


def checkpoint_temporal_partition_kind(payload):
    """Return the exact accepted temporal-partition kind before native restart mutation."""
    contract = _decode_contract(payload)
    rows = contract["temporal_partition"]
    if (
        not isinstance(rows, list)
        or not rows
        or not isinstance(rows[0], list)
        or len(rows[0]) != 7
        or rows[0][0] != "summary"
        or rows[0][1] not in {"global", "cell_local"}
    ):
        raise ValueError("restart: AMR temporal-partition contract has an invalid summary")
    for row in rows[1:]:
        if not isinstance(row, list) or len(row) != 3 or row[0] != "rung":
            raise ValueError("restart: AMR temporal-partition contract has an invalid rung row")
    return rows[0][1]


def preflight_contract(sim, payload):
    """Authenticate shape and static provenance before the native restart transaction."""
    import numpy as np

    required = {
        "amr_accepted_contract",
        "program_accepted_state",
        "regrid_count",
        "topology_epoch",
    }
    missing = sorted(required.difference(getattr(payload, "files", payload.keys())))
    if missing:
        raise ValueError("restart: AMR checkpoint lacks accepted-state keys %r" % missing)
    contract = _decode_contract(payload)
    current = contract_for(sim)
    if contract != current:
        mismatched = sorted(key for key in _PREFLIGHT_KEYS if contract.get(key) != current.get(key))
        if mismatched:
            raise ValueError(
                "restart: AMR static accepted-state provenance differs from the installed "
                "composition (mismatched sections: %r)" % mismatched
            )
    state = np.asarray(payload["program_accepted_state"])
    if state.dtype != np.dtype("uint8") or state.ndim != 1:
        raise ValueError("restart: AMR Program accepted state must be a uint8 vector")
    if bool(sim.installed_program_hash()) != bool(state.size):
        raise ValueError(
            "restart: compiled AMR Program requires a non-empty accepted state; a native route "
            "must not carry one"
        )
    regrid_count = int(payload["regrid_count"])
    topology_epoch = int(payload["topology_epoch"])
    if regrid_count < 0 or topology_epoch < 0:
        raise ValueError("restart: AMR regrid count/topology epoch must be non-negative")
    return state.tobytes(), regrid_count, topology_epoch


def _validate_interface_ledger_against_live_hierarchy(sim, contract):
    epoch = int(sim.checkpoint_topology_epoch())
    levels = int(sim.n_levels())
    blocks = int(sim.n_blocks())
    for row in contract["interface_ledger"]["entries"]:
        if len(row) != 31:
            raise ValueError("restart: restored AMR interface-flux audit has an invalid native row")
        coarse_level, fine_level = int(row[2]), int(row[3])
        left_block, right_block = int(row[21]), int(row[22])
        graph_identity, rate_identity, application_identity = row[28:31]
        if (
            int(row[1]) != epoch
            or coarse_level < 0
            or fine_level != coarse_level + 1
            or fine_level >= levels
            or left_block < 0
            or right_block < 0
            or left_block >= blocks
            or right_block >= blocks
            or row[27] != "resolved"
            or graph_identity != sim.installed_program_hash()
            or not rate_identity
            or not application_identity
        ):
            raise ValueError(
                "restart: restored AMR interface-flux audit is outside the live hierarchy"
            )


def validate_restored_contract(sim, payload):
    """Validate the dynamic contract after the opaque Program image is installed transactionally."""
    contract = _decode_contract(payload)
    current = contract_for(sim)
    if contract != current:
        mismatched = sorted(key for key in _CONTRACT_KEYS if contract.get(key) != current.get(key))
        raise ValueError(
            "restart: restored AMR accepted-state image differs from its authenticated contract "
            "(mismatched sections: %r)" % mismatched
        )
    _validate_interface_ledger_against_live_hierarchy(sim, current)


def validate_regridded_contract(sim, payload, receipt):
    """Validate the weaker but explicit accepted-state contract after one restart regrid."""
    required = {
        "schema_version",
        "policy_identity",
        "changed",
        "accepted_time",
        "accepted_macro_step",
        "before",
        "after",
        "accepted_contract_identity_before",
        "accepted_contract_identity_after",
        "history_consensus_identity_before",
        "history_consensus_identity_after",
        "composite_integrals_before",
        "composite_integrals_after",
        "field_manifest_identity_before",
        "field_manifest_identity_after",
        "field_manifest_before",
        "field_manifest_after",
        "field_recompute_witness",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise TypeError("restart: RegridOnRestart receipt has an invalid exact schema")
    if receipt["schema_version"] != 3 or type(receipt["changed"]) is not bool:
        raise ValueError("restart: RegridOnRestart receipt version or changed flag is invalid")
    accepted_time = float(payload["t"])
    accepted_step = int(payload["macro_step"])
    if (
        float(sim.time()) != accepted_time
        or int(sim.macro_step()) != accepted_step
        or receipt["accepted_time"] != accepted_time
        or receipt["accepted_macro_step"] != accepted_step
    ):
        raise ValueError(
            "restart: RegridOnRestart changed the accepted physical clock or macro-step"
        )

    levels = int(sim.n_levels())
    current_after = restart_topology_image(sim)
    if receipt["after"] != current_after:
        raise ValueError(
            "restart: RegridOnRestart receipt differs from the live transformed hierarchy"
        )
    before = receipt["before"]
    after = receipt["after"]
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise TypeError("restart: RegridOnRestart topology receipt must contain mappings")
    if "topology_identity" not in before or "topology_identity" not in after:
        raise ValueError("restart: RegridOnRestart topology receipt lacks its structural identity")
    # A scientific regrid attempt advances its audit counters even when tagging reproduces the
    # exact same boxes and owner map.  ``changed`` reports a structural hierarchy change, not merely
    # that the restart policy executed.
    changed = before["topology_identity"] != after["topology_identity"]
    if receipt["changed"] is not changed:
        raise ValueError("restart: RegridOnRestart receipt has an inconsistent changed flag")

    from pops.identity import Identity, make_identity

    recorded = _decode_contract(payload)
    transformed = contract_for(sim)
    expected_before_identity = make_identity("restart-accepted-contract", recorded).token
    expected_after_identity = make_identity("restart-accepted-contract", transformed).token
    if (
        receipt["accepted_contract_identity_before"] != expected_before_identity
        or receipt["accepted_contract_identity_after"] != expected_after_identity
    ):
        raise ValueError(
            "restart: RegridOnRestart accepted-contract audit identity differs from "
            "the recorded or transformed Program image"
        )
    # These are phase-local all-rank witnesses, not a bitwise-continuity assertion across a
    # topology change.  A scientific regrid changes the dense level-domain encoding and
    # interpolates newly refined cells, so equality before/after would be the wrong invariant.
    # Conservation of the accepted solution is checked independently below from the composite
    # per-block component integrals.
    for key in (
        "history_consensus_identity_before",
        "history_consensus_identity_after",
    ):
        identity = Identity.from_token(receipt[key])
        if identity.domain != "restart-history-image" or identity.schema_version != 1:
            raise ValueError(
                "restart: RegridOnRestart phase-local history consensus identity has the "
                "wrong domain or schema version"
            )
    for key in ("field_manifest_identity_before", "field_manifest_identity_after"):
        identity = Identity.from_token(receipt[key])
        if identity.domain != "restart-field-provider-manifest" or identity.schema_version != 1:
            raise ValueError(
                "restart: RegridOnRestart field-provider manifest witness has the wrong domain "
                "or schema version"
            )
    before_field_manifest = receipt["field_manifest_before"]
    after_field_manifest = receipt["field_manifest_after"]
    if (
        not isinstance(before_field_manifest, list)
        or not isinstance(after_field_manifest, list)
        or any(not isinstance(row, list) for row in before_field_manifest + after_field_manifest)
    ):
        raise TypeError("restart: RegridOnRestart field-provider manifest witness is invalid")
    if _field_provider_contract_rows(before_field_manifest) != recorded["field_providers"]:
        raise ValueError(
            "restart: RegridOnRestart recorded field-provider semantics changed before regrid"
        )
    live_field_manifest = _rows(sim.field_provider_checkpoint_manifest())
    if after_field_manifest != live_field_manifest:
        raise ValueError(
            "restart: RegridOnRestart transformed field-provider manifest differs from the live image"
        )
    for rows, topology in ((before_field_manifest, before), (after_field_manifest, after)):
        if any(row[6] != str(topology["topology_epoch"]) or row[8] != "materialized" for row in rows):
            raise ValueError(
                "restart: RegridOnRestart field-provider witness has invalid topology or materialization"
            )
    expected_field_before = make_identity(
        "restart-field-provider-manifest",
        {"schema_version": 1, "providers": before_field_manifest},
    ).token
    expected_field_after = make_identity(
        "restart-field-provider-manifest",
        {"schema_version": 1, "providers": after_field_manifest},
    ).token
    if (
        receipt["field_manifest_identity_before"] != expected_field_before
        or receipt["field_manifest_identity_after"] != expected_field_after
    ):
        raise ValueError(
            "restart: RegridOnRestart field-provider manifest witness differs from the recorded "
            "or transformed image"
        )
    witness = receipt["field_recompute_witness"]
    if not isinstance(witness, list) or any(
        not isinstance(row, list) or len(row) != 16 for row in witness
    ):
        raise TypeError("restart: RegridOnRestart field recompute witness has an invalid schema")
    slots = [str(row[1]) for row in transformed["field_providers"]]
    if len({row[0] for row in witness}) != len(witness) or sorted(
        row[0] for row in witness
    ) != sorted(slots):
        raise ValueError(
            "restart: RegridOnRestart field recompute witness differs from the live provider order"
        )
    if any(row[5] != "solved" for row in witness):
        raise ValueError("restart: RegridOnRestart field recompute did not publish a solved value")
    manifest_by_slot = {row[1]: row for row in after_field_manifest}
    accepted_time_bits = int.from_bytes(struct.pack("!d", accepted_time), "big")
    for row in witness:
        manifest = manifest_by_slot[row[0]]
        try:
            dt_bits = int(row[14])
            time_bits = int(row[15])
            dt = struct.unpack("!d", dt_bits.to_bytes(8, "big"))[0]
        except (OverflowError, ValueError, struct.error):
            raise ValueError(
                "restart: RegridOnRestart field recompute point has invalid binary64 evidence"
            ) from None
        if (
            row[1] != manifest[4]
            or row[2] != manifest[3]
            or row[3] != str(after["topology_epoch"])
            or row[4] != manifest[7]
            or row[6] != "pops.amr.restart-regrid.accepted"
            or row[7] != str(accepted_step)
            or row[8:14] != ["0", "0", "0", "0", "0", "1"]
            or not _valid_field_recompute_dt(dt, accepted_step)
            or time_bits != accepted_time_bits
        ):
            raise ValueError(
                "restart: RegridOnRestart field recompute point differs from its accepted authority"
            )
    expected_program_state = recorded["program_state"]
    if transformed["program_state"] != expected_program_state:
        raise ValueError("restart: RegridOnRestart changed the installed Program authority")
    if transformed["ledger"]["transaction_depth"] != 0:
        raise ValueError("restart: RegridOnRestart left a conservative ledger transaction active")
    if transformed["interface_ledger"]["transaction_depth"] != 0:
        raise ValueError(
            "restart: RegridOnRestart left an interface-flux ledger transaction active"
        )

    level_clocks = [row for row in transformed["clocks"] if row and row[0] == "level"]
    if len(level_clocks) != levels:
        raise ValueError("restart: RegridOnRestart did not requalify every active AMR level clock")
    for level, row in enumerate(level_clocks):
        if (
            len(row) != 6
            or int(row[1]) != level
            or int(row[2]) != accepted_step
            or (int(row[3]), int(row[4])) != (0, 1)
            or float(row[5]) != accepted_time
        ):
            raise ValueError(
                "restart: RegridOnRestart produced a non-accepted or misqualified level clock"
            )
    history_slots = {}
    history_publication = {}
    for row in transformed["history_qualifications"]:
        if len(row) != 13 or int(row[7]) != levels:
            raise ValueError(
                "restart: RegridOnRestart did not requalify a history over all active levels"
            )
        depth = int(row[6])
        level = int(row[8])
        slot = int(row[9])
        dt_bits = int(row[10])
        initialized = int(row[11])
        fill_count = int(row[12])
        if (
            depth < 2
            or level < 0
            or level >= levels
            or slot < 0
            or slot >= depth
            or dt_bits < 0
            or dt_bits >= 1 << 64
            or initialized not in (0, 1)
            or fill_count < 0
            or fill_count > depth
            or bool(initialized) != (fill_count > 0)
        ):
            raise ValueError(
                "restart: RegridOnRestart produced invalid history-slot provenance"
            )
        outgoing_dt = struct.unpack("!d", dt_bits.to_bytes(8, "big"))[0]
        if not math.isfinite(outgoing_dt) or (
            fill_count == 0 and outgoing_dt != 0.0
        ) or (fill_count > 0 and outgoing_dt <= 0.0):
            raise ValueError(
                "restart: RegridOnRestart produced invalid history-slot outgoing dt"
            )
        descriptor = tuple(row[:8])
        publication_key = (descriptor, level)
        publication = (initialized, fill_count)
        if publication_key in history_publication and history_publication[publication_key] != publication:
            raise ValueError(
                "restart: RegridOnRestart produced inconsistent history publication metadata"
            )
        history_publication[publication_key] = publication
        coordinates = history_slots.setdefault(descriptor, set())
        if (level, slot) in coordinates:
            raise ValueError(
                "restart: RegridOnRestart duplicated history-slot provenance"
            )
        coordinates.add((level, slot))
    for descriptor, coordinates in history_slots.items():
        depth = int(descriptor[6])
        expected = {(level, slot) for level in range(levels) for slot in range(depth)}
        if coordinates != expected:
            raise ValueError(
                "restart: RegridOnRestart omitted history-slot provenance"
            )
    if (
        transformed["ledger"]["accepted_entries"]
        or transformed["interface_ledger"]["accepted_entries"]
        or transformed["synchronization"]
    ):
        raise ValueError(
            "restart: RegridOnRestart exposed pre-transform flux/interface or "
            "synchronization reports"
        )


__all__ = [
    "contract_for",
    "encode_contract",
    "preflight_contract",
    "restart_topology_image",
    "validate_regridded_contract",
    "validate_restored_contract",
]
