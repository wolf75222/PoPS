"""Canonical accepted-state contract for strict AMR checkpoint/restart."""
from __future__ import annotations

import json


_SCHEMA = 1
_GUARANTEE = "bit_identical_accepted_state"


def _rows(values):
    return [list(map(str, row)) for row in values]


def contract_for(sim):
    """Return the audit-readable part of the native accepted-state image."""
    ratios = [int(value) for value in sim.checkpoint_temporal_ratios()]
    return {
        "schema_version": _SCHEMA,
        "guarantee": _GUARANTEE,
        "program_state": "compiled" if sim.installed_program_hash() else "native_none",
        "ledger": {"accepted_entries": 0, "transaction_depth": 0},
        "history_qualifications": _rows(sim.program_accepted_state_manifest()),
        "level_relations": [
            {
                "parent": parent,
                "child": parent + 1,
                "temporal_ratio": ratio,
                "remainder_policy": "integral_only",
            }
            for parent, ratio in enumerate(ratios)
        ],
        "transfer_routes": _rows(sim.checkpoint_transfer_routes()),
    }


def encode_contract(sim):
    return json.dumps(contract_for(sim), sort_keys=True, separators=(",", ":"), allow_nan=False)


def preflight_contract(sim, payload):
    """Authenticate shape and static provenance before the native restart transaction."""
    import numpy as np
    from pops._manifest_protocol import strict_json_loads

    required = {
        "amr_accepted_contract", "program_accepted_state", "regrid_count", "topology_epoch",
    }
    missing = sorted(required.difference(getattr(payload, "files", payload.keys())))
    if missing:
        raise ValueError("restart: AMR checkpoint lacks accepted-state keys %r" % missing)
    contract = strict_json_loads(str(payload["amr_accepted_contract"]),
                                 where="AMR accepted-state contract")
    if not isinstance(contract, dict):
        raise TypeError("restart: AMR accepted-state contract must be a mapping")
    current = contract_for(sim)
    if contract != current:
        mismatched = sorted(
            key for key in set(contract).union(current) if contract.get(key) != current.get(key))
        raise ValueError(
            "restart: AMR accepted-state provenance differs from the installed owners, spaces, "
            "level relations, or transfer plans (mismatched sections: %r)" % mismatched)
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


__all__ = ["contract_for", "encode_contract", "preflight_contract"]
