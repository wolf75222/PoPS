"""Source-only real public AMR plans; native mutation seams are explicit spies."""
from copy import deepcopy
import pytest
from pops.amr import AMRExecution, AMRClockRelation, AcceptedHaloPreparation
from pops.amr._execution_contract import runtime_execution_data, validate_execution_data
from pops.runtime._runtime_authorities import install_runtime_authorities
from tests.python.unit.codegen.test_install_tagging_authority import metadata_platform, _install
from tests.python.unit.codegen._typed_artifact_fixture import artifact_fixture


class Engine:
    def __init__(self): self.calls = []
    def set_temporal_relations(self, *args): self.calls.append(("clock", args))


@pytest.fixture
def spies(monkeypatch):
    for name in ("_install_boundary_authorities", "_install_amr_provider_authorities"):
        monkeypatch.setattr("pops.runtime._runtime_authorities." + name,
                            lambda engine, plan, kind=name: engine.calls.append((kind, plan)))
    monkeypatch.setattr("pops.runtime._runtime_mesh_lowering.flow_bootstrap_tagging",
                        lambda engine, *args, **kwargs: engine.calls.append(("bootstrap", kwargs)))


@pytest.mark.parametrize("execution", (
    AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    AMRExecution.synchronous(accepted_halo=AcceptedHaloPreparation((1, 2))),
    AMRExecution.subcycled((AMRClockRelation(0, 1, 2),), accepted_halo=AcceptedHaloPreparation(1)),
))
def test_real_install_plan_preserves_execution_and_installs_clock(metadata_platform, spies, execution):
    install = _install(artifact_fixture(target="amr_system", execution=execution))
    install.verify()
    engine = Engine()
    install_runtime_authorities(engine, install)
    data = dict(engine._amr_execution_authority)
    assert data["schema_version"] == execution.to_data()["schema_version"]
    assert data["mode"] == execution.mode
    if execution.accepted_halo is not None:
        assert dict(data["accepted_halo"]) == execution.accepted_halo.to_data()
        with pytest.raises(TypeError): data["accepted_halo"]["cells"] = 8
    else:
        assert data == execution.to_data()
    assert [row[0] for row in engine.calls] == [
        "_install_boundary_authorities", "_install_amr_provider_authorities", "clock", "bootstrap"]


@pytest.mark.parametrize("change", (
    lambda d: d.update(schema_version=4),
    lambda d: d.update(schema_version=3.0),
    lambda d: d.pop("accepted_halo"),
    lambda d: d["accepted_halo"].update(schema_version=True),
    lambda d: d["accepted_halo"].update(cells=(True, 1)),
    lambda d: d["accepted_halo"].update(cells=(1,)),
    lambda d: d["accepted_halo"].update(cells=(1 << 31, 1)),
    lambda d: d["accepted_halo"].update(point_authority="user_time"),
    lambda d: d["accepted_halo"].update(extra=1),
))
def test_bad_execution_refuses_before_any_mutation(metadata_platform, spies, change):
    from types import SimpleNamespace
    install = _install(artifact_fixture(target="amr_system"))
    data = AMRExecution.synchronous(accepted_halo=AcceptedHaloPreparation((1, 1))).to_data()
    change(data)
    # Adversarial open protocol over a real plan; no fake Native acceptance.
    view = SimpleNamespace(artifact=install.artifact, resolved_hierarchy=install.resolved_hierarchy,
                           amr_execution=SimpleNamespace(runtime_execution_data=lambda: data))
    engine = Engine()
    with pytest.raises((TypeError, ValueError)): install_runtime_authorities(engine, view)
    assert engine.calls == []
    assert not hasattr(engine, "_amr_execution_authority")


def test_protocol_snapshot_detects_reused_mutable_dict_and_no_alias():
    from types import SimpleNamespace
    data = AMRExecution.synchronous().to_data()
    def drifting():
        data["relations"] = []
        data["mode"] = "subcycled" if data["mode"] == "synchronous" else "synchronous"
        return data
    with pytest.raises(TypeError): runtime_execution_data(SimpleNamespace(runtime_execution_data=drifting))
    good = AMRExecution.synchronous(accepted_halo=AcceptedHaloPreparation()).to_data()
    snapshot = validate_execution_data(good)
    good["accepted_halo"]["cells"] = 5
    assert snapshot["accepted_halo"]["cells"] == 1
