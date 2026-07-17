"""Canonical accepted-state contract for strict AMR checkpoint/restart."""
from __future__ import annotations

import json


_SCHEMA = 2
_GUARANTEE = "bit_identical_accepted_state"
_CONTRACT_KEYS = {
    "schema_version", "guarantee", "program_state", "ledger", "clocks",
    "synchronization", "history_qualifications", "level_relations", "transfer_routes",
}
_PREFLIGHT_KEYS = {
    "schema_version", "guarantee", "program_state", "level_relations", "transfer_routes",
}


def _rows(values):
    return [list(map(str, row)) for row in values]


def contract_for(sim):
    """Return the audit-readable part of the native accepted-state image."""
    relation_rows = [list(map(str, row)) for row in sim.checkpoint_temporal_relations()]
    relations = []
    for row in relation_rows:
        if len(row) != 5:
            raise ValueError("native AMR temporal relation report has an invalid row")
        parent, child, numerator, denominator, remainder = row
        relations.append({
            "parent": int(parent), "child": int(child),
            "temporal_ratio": {
                "numerator": int(numerator), "denominator": int(denominator),
            },
            "remainder_policy": remainder,
        })
    flux_ledger = _rows(sim.program_flux_ledger_manifest())
    return {
        "schema_version": _SCHEMA,
        "guarantee": _GUARANTEE,
        "program_state": "compiled" if sim.installed_program_hash() else "native_none",
        "ledger": {
            "accepted_entries": len(flux_ledger), "transaction_depth": 0,
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
        str(payload["amr_accepted_contract"]), where="AMR accepted-state contract")
    if not isinstance(contract, dict) or set(contract) != _CONTRACT_KEYS:
        raise TypeError("restart: AMR accepted-state contract has an invalid exact schema")
    return contract


def preflight_contract(sim, payload):
    """Authenticate shape and static provenance before the native restart transaction."""
    import numpy as np
    required = {
        "amr_accepted_contract", "program_accepted_state", "regrid_count", "topology_epoch",
    }
    missing = sorted(required.difference(getattr(payload, "files", payload.keys())))
    if missing:
        raise ValueError("restart: AMR checkpoint lacks accepted-state keys %r" % missing)
    contract = _decode_contract(payload)
    current = contract_for(sim)
    if contract != current:
        mismatched = sorted(
            key for key in _PREFLIGHT_KEYS if contract.get(key) != current.get(key))
        if mismatched:
            raise ValueError(
                "restart: AMR static accepted-state provenance differs from the installed "
                "composition (mismatched sections: %r)" % mismatched)
    state = np.asarray(payload["program_accepted_state"])
    if state.dtype != np.dtype("uint8") or state.ndim != 1:
        raise ValueError("restart: AMR Program accepted state must be a uint8 vector")
    if bool(sim.installed_program_hash()) != bool(state.size):
        raise ValueError(
            "restart: compiled AMR Program requires a non-empty accepted state; a native route "
            "must not carry one")
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
        mismatched = sorted(
            key for key in _CONTRACT_KEYS if contract.get(key) != current.get(key))
        raise ValueError(
            "restart: restored AMR accepted-state image differs from its authenticated contract "
            "(mismatched sections: %r)" % mismatched)


__all__ = [
    "contract_for", "encode_contract", "preflight_contract", "validate_restored_contract",
]
