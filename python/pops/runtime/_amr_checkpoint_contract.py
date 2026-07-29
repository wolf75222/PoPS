"""Canonical accepted-state contract for strict AMR checkpoint/restart."""

from __future__ import annotations

from collections.abc import Mapping
import json

from pops.identity import make_identity


_SCHEMA = 4
_GUARANTEE = "bit_identical_accepted_state"
_CONTRACT_KEYS = {
    "schema_version",
    "guarantee",
    "program_state",
    "ledger",
    "interface_ledger",
    "clocks",
    "synchronization",
    "history_qualifications",
    "level_relations",
    "transfer_routes",
}
_PREFLIGHT_KEYS = {
    "schema_version",
    "guarantee",
    "program_state",
    "level_relations",
    "transfer_routes",
}


def _rows(values):
    return [list(map(str, row)) for row in values]


def restart_topology_image(sim):
    """Return the compact identity of one accepted AMR hierarchy."""
    levels = int(sim.n_levels())
    boxes = [[int(value) for value in box] for box in sim.patch_boxes()]
    owners = [
        [int(rank) for rank in sim.level_owner_ranks(level)] for level in range(levels)
    ]
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
        "synchronization": _rows(sim.program_sync_manifest()),
        "history_qualifications": _rows(sim.program_accepted_state_manifest()),
        "level_relations": relations,
        "transfer_routes": _rows(sim.checkpoint_transfer_routes()),
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
        if len(row) != 28:
            raise ValueError(
                "restart: restored AMR interface-flux audit has an invalid native row"
            )
        coarse_level, fine_level = int(row[2]), int(row[3])
        left_block, right_block = int(row[21]), int(row[22])
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
        "history_identity_before",
        "history_identity_after",
        "composite_integrals_before",
        "composite_integrals_after",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise TypeError("restart: RegridOnRestart receipt has an invalid exact schema")
    if receipt["schema_version"] != 2 or type(receipt["changed"]) is not bool:
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
    for key in ("history_identity_before", "history_identity_after"):
        identity = Identity.from_token(receipt[key])
        if identity.domain != "restart-history-image" or identity.schema_version != 1:
            raise ValueError(
                "restart: RegridOnRestart history rematerialization identity has the wrong "
                "domain or schema version"
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
    for row in transformed["history_qualifications"]:
        if len(row) != 8 or int(row[7]) != levels:
            raise ValueError(
                "restart: RegridOnRestart did not requalify a history over all active levels"
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
