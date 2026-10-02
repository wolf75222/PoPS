"""Genuine public Source plan; mutation-counting sinks are NOT Native qualification."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import json
import numpy as np
import pytest
import pops
from pops.amr import AMRExecution
from tests.python.support.tag_selection_case import build
from pops.runtime import _runtime_authorities as authorities
from pops.runtime._amr_checkpoint_contract import _decode_contract
from tests.python.unit.runtime.test_amr_checkpoint_contract import _payload

ROOT = Path(__file__).resolve().parents[2]

@pytest.fixture(scope="module")
def genuine_plan():
    assert Path(pops.__file__).resolve() == ROOT / "python/pops/__init__.py"
    assert not any(n == "_pops" or n.endswith("._pops") for n in sys.modules)
    case, layout = build((8,12), 0, "linear")
    return pops.resolve(pops.validate(case), layout=layout)


def install(data, plan, monkeypatch):
    calls = []
    monkeypatch.setattr(authorities, "_install_boundary_authorities", lambda *_: calls.append("boundary"))
    monkeypatch.setattr(authorities, "_install_amr_provider_authorities", lambda *_: calls.append("providers"))
    # Real resolved layout/hierarchy/execution; omit bootstrap at this isolated seam.
    projected = SimpleNamespace(artifact=SimpleNamespace(layout_plan=plan.layout_plan, plan=plan),
        amr_execution=SimpleNamespace(runtime_execution_data=lambda: deepcopy(data)),
        resolved_hierarchy=plan.resolved_hierarchy, bootstrap_plan=None)
    engine = SimpleNamespace(set_temporal_relations=lambda *args: calls.append(("relations", args)))
    return calls, engine, projected


@pytest.mark.parametrize("version", (2,3))
def test_default_and_optin_install_actual_plan(genuine_plan, monkeypatch, version):
    data = (AMRExecution.synchronous().to_data() if version == 2 else genuine_plan.amr_execution.to_data())
    assert data["schema_version"] == version
    if version == 2:
        assert data == {"schema_version":2,"authority_type":"amr_execution","mode":"synchronous","relations":[]}
    calls, engine, projected = install(data, genuine_plan, monkeypatch)
    authorities.install_runtime_authorities(engine, projected)
    assert calls[:2] == ["boundary", "providers"]
    assert calls[2] == ("relations", ([1], [1], ["integral_only"]))
    assert engine._amr_execution_authority["schema_version"] == version
    if version == 3:
        assert engine._amr_execution_authority["accepted_halo"] == data["accepted_halo"]


@pytest.mark.parametrize("attack", ("version", "float-version", "missing-halo", "extra", "effect", "float-effect-version", "bool-width", "rank", "authority", "sync-relations", "bool-parent", "bool-child"))
def test_bad_execution_refuses_before_all_mutating_callbacks(genuine_plan, monkeypatch, attack):
    data = deepcopy(genuine_plan.amr_execution.to_data())
    if attack == "version": data["schema_version"] = 4
    elif attack == "float-version": data["schema_version"] = 3.
    elif attack == "missing-halo": del data["accepted_halo"]
    elif attack == "extra": data["other_effect"] = {}
    elif attack == "effect": data["accepted_halo"]["effect"] = "publish_without_preparation"
    elif attack == "float-effect-version": data["accepted_halo"]["schema_version"] = 1.
    elif attack == "bool-width": data["accepted_halo"]["cells"] = True
    elif attack == "rank": data["accepted_halo"]["cells"] = (1,1,1)
    elif attack == "authority": data["accepted_halo"]["point_authority"] = "user_clock"
    else:
        data["relations"] = [{"parent_level":0,"child_level":1,"temporal_ratio":{"numerator":1,"denominator":1},"remainder_policy":"integral_only"}]
        if attack != "sync-relations":
            data["mode"] = "subcycled"
            data["relations"][0]["parent_level" if attack == "bool-parent" else "child_level"] = (False if attack == "bool-parent" else True)
    calls, engine, projected = install(data, genuine_plan, monkeypatch)
    with pytest.raises((TypeError,ValueError)):
        authorities.install_runtime_authorities(engine, projected)
    assert calls == []
    assert not hasattr(engine, "_amr_execution_authority")


@pytest.mark.parametrize("version", (8,9))
def test_checkpoint_exact_schema_and_optional_halo(version):
    payload = _payload()
    contract = json.loads(str(payload["amr_accepted_contract"]))
    contract["schema_version"] = version
    if version == 9:
        contract["accepted_halo"] = [["pops.amr.accepted-halo-preparation@1","candidate_accepted_clock","all_state_components","1","1"]]
    payload["amr_accepted_contract"] = np.array(json.dumps(contract))
    assert _decode_contract(payload) == contract
    contract["schema_version"] = float(version)
    payload["amr_accepted_contract"] = np.array(json.dumps(contract))
    with pytest.raises(TypeError): _decode_contract(payload)


@pytest.mark.parametrize("attack", ("unknown-schema", "missing", "foreign-clock", "foreign-components", "zero", "leading-zero", "bool", "extra-row"))
def test_checkpoint_halo_closed_point_extent_contract(attack):
    payload = _payload()
    contract = json.loads(str(payload["amr_accepted_contract"]))
    contract["schema_version"] = 9
    row = ["pops.amr.accepted-halo-preparation@1","candidate_accepted_clock","all_state_components","1","1"]
    contract["accepted_halo"] = [row]
    if attack == "unknown-schema": contract["schema_version"] = 10
    elif attack == "missing": del contract["accepted_halo"]
    elif attack == "foreign-clock": row[1] = "user_clock"
    elif attack == "foreign-components": row[2] = "first_state_component"
    elif attack == "zero": row[3] = "0"
    elif attack == "leading-zero": row[3] = "01"
    elif attack == "bool": row[3] = True
    elif attack == "extra-row": contract["accepted_halo"].append(row.copy())
    payload["amr_accepted_contract"] = np.array(json.dumps(contract))
    with pytest.raises(TypeError): _decode_contract(payload)
