"""ADC-678 strict AMR accepted-state and topology checkpoint contracts."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pops.runtime._amr_checkpoint_contract import (
    contract_for,
    encode_contract,
    preflight_contract,
    validate_restored_contract,
)
from pops.runtime._amr_checkpoint_topology import owner_ranks_for_boxes


class _Payload(dict):
    @property
    def files(self):
        return list(self)


class _Sim:
    program_hash = "ab" * 32

    def installed_program_hash(self):
        return self.program_hash

    def checkpoint_temporal_relations(self):
        return [[0, 1, 2, 1, "integral_only"], [1, 2, 3, 1, "integral_only"]]

    def checkpoint_transfer_routes(self):
        return [["fluid.U", "prolong", "route.u", "provider.u", "kernel.linear",
                 "cell.conservative", "cell", "conservative", "dense", "prolong",
                 "2", "2,2", "2", "2"]]

    def program_accepted_state_manifest(self):
        return [["rhs", "program.block.0", "fluid.U", "cell.conservative",
                 "clock.macro", "dense.linear", "2", "3"]]

    def program_clock_manifest(self):
        return [["level", "0", "4", "0", "1", "0.4"],
                ["logical", "clock.macro", "4"]]

    def program_flux_ledger_manifest(self):
        return [["program.block.0", "fluid.U", "rate.7", "physical_flux", "1",
                 "4", "1", "2", "1", "2", "x_plus", "0.125", "0.05"]]

    def program_sync_manifest(self):
        return [["0", "1", "0", "reflux", "4", "1", "1"],
                ["0", "1", "0", "average_down", "4", "1", "1"]]


def _payload(sim=None):
    sim = sim or _Sim()
    return _Payload({
        "amr_accepted_contract": np.array(encode_contract(sim)),
        "program_accepted_state": np.array([1, 2, 3], dtype=np.uint8),
        "regrid_count": np.array(4),
        "topology_epoch": np.array(7, dtype=np.uint64),
    })


def test_contract_names_guarantee_relations_qualified_histories_and_transfer_plans():
    contract = contract_for(_Sim())
    assert contract["guarantee"] == "bit_identical_accepted_state"
    assert contract["ledger"]["accepted_entries"] == 1
    assert contract["ledger"]["transaction_depth"] == 0
    assert contract["ledger"]["entries"][0][8:10] == ["1", "2"]
    assert contract["level_relations"] == [
        {"parent": 0, "child": 1,
         "temporal_ratio": {"numerator": 2, "denominator": 1},
         "remainder_policy": "integral_only"},
        {"parent": 1, "child": 2,
         "temporal_ratio": {"numerator": 3, "denominator": 1},
         "remainder_policy": "integral_only"},
    ]
    assert contract["history_qualifications"][0][1:4] == [
        "program.block.0", "fluid.U", "cell.conservative"]
    assert contract["transfer_routes"][0][2:5] == [
        "route.u", "provider.u", "kernel.linear"]
    assert contract["clocks"][1] == ["logical", "clock.macro", "4"]
    assert [row[3] for row in contract["synchronization"]] == [
        "reflux", "average_down"]


def test_preflight_returns_exact_native_payload_and_counters():
    state, regrids, epoch = preflight_contract(_Sim(), _payload())
    assert state == b"\x01\x02\x03"
    assert (regrids, epoch) == (4, 7)


@pytest.mark.parametrize("mutation", ["ratio", "route", "guarantee"])
def test_preflight_refuses_any_static_provenance_mismatch(mutation):
    payload = _payload()
    data = json.loads(str(payload["amr_accepted_contract"]))
    if mutation == "ratio":
        data["level_relations"][0]["temporal_ratio"]["numerator"] = 4
    elif mutation == "route":
        data["transfer_routes"][0][3] = "provider.other"
    else:
        data["guarantee"] = "regrid_on_restart"
    payload["amr_accepted_contract"] = np.array(json.dumps(data))
    with pytest.raises(ValueError, match="provenance differs"):
        preflight_contract(_Sim(), payload)


@pytest.mark.parametrize("section", ["history_qualifications", "clocks", "ledger", "synchronization"])
def test_dynamic_contract_is_checked_after_the_opaque_state_is_restored(section):
    payload = _payload()
    data = json.loads(str(payload["amr_accepted_contract"]))
    if section == "history_qualifications":
        data[section][0][1] = "program.block.1"
    elif section == "ledger":
        data[section]["accepted_entries"] += 1
    else:
        data[section].append(["tampered"])
    payload["amr_accepted_contract"] = np.array(json.dumps(data))
    preflight_contract(_Sim(), payload)
    with pytest.raises(ValueError, match="restored AMR accepted-state image differs"):
        validate_restored_contract(_Sim(), payload)


def test_native_route_requires_no_program_blob_and_compiled_route_requires_one():
    compiled = _payload()
    compiled["program_accepted_state"] = np.array([], dtype=np.uint8)
    with pytest.raises(ValueError, match="requires a non-empty accepted state"):
        preflight_contract(_Sim(), compiled)

    native_sim = _Sim()
    native_sim.program_hash = ""
    native = _payload(native_sim)
    native["program_accepted_state"] = np.array([], dtype=np.uint8)
    assert preflight_contract(native_sim, native)[0] == b""


def test_topology_owner_alignment_is_level_local_and_strict():
    payload = {"dmap_1": np.array([2, 0]), "dmap_2": np.array([1])}
    boxes = [(1, 0, 0, 3, 3), (1, 4, 4, 7, 7), (2, 2, 2, 5, 5)]
    assert owner_ranks_for_boxes(payload, boxes, 3) == [2, 0, 1]
    with pytest.raises(ValueError, match="truncated"):
        owner_ranks_for_boxes(payload, boxes + [(2, 6, 6, 7, 7)], 3)
    with pytest.raises(ValueError, match="lacks owner-rank map"):
        owner_ranks_for_boxes({}, boxes[:1], 3)
