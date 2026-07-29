"""Canonical accepted-state contract for strict AMR checkpoint/restart."""

from __future__ import annotations

import json

from pops.identity import make_identity


_SCHEMA = 2
_GUARANTEE = "bit_identical_accepted_state"
_CONTRACT_KEYS = {
    "schema_version",
    "guarantee",
    "program_state",
    "ledger",
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
    return {
        "schema_version": _SCHEMA,
        "guarantee": _GUARANTEE,
        "program_state": "compiled" if sim.installed_program_hash() else "native_none",
        "ledger": {
            "accepted_entries": len(flux_ledger),
            "transaction_depth": 0,
            "entries": flux_ledger,
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
        "composite_integrals_before",
        "composite_integrals_after",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise TypeError("restart: RegridOnRestart receipt has an invalid exact schema")
    if receipt["schema_version"] != 1 or type(receipt["changed"]) is not bool:
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
    changed = receipt["before"] != receipt["after"]
    if receipt["changed"] is not changed:
        raise ValueError("restart: RegridOnRestart receipt has an inconsistent changed flag")

    transformed = contract_for(sim)
    expected_program_state = _decode_contract(payload)["program_state"]
    if transformed["program_state"] != expected_program_state:
        raise ValueError("restart: RegridOnRestart changed the installed Program authority")
    if transformed["ledger"]["transaction_depth"] != 0:
        raise ValueError("restart: RegridOnRestart left a conservative ledger transaction active")

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
    if transformed["ledger"]["accepted_entries"] or transformed["synchronization"]:
        raise ValueError(
            "restart: RegridOnRestart exposed pre-transform flux or synchronization reports"
        )


__all__ = [
    "contract_for",
    "encode_contract",
    "preflight_contract",
    "restart_topology_image",
    "validate_regridded_contract",
    "validate_restored_contract",
]
