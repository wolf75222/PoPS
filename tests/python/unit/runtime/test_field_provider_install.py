"""Both native engine adapters preserve the same authenticated component ABI."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from pops.runtime._amr_system_install import _PreparedAmrFieldSolverInstall
from pops.runtime._system_unified_install import _PreparedSystemFieldSolverInstall


@pytest.fixture(params=[_PreparedSystemFieldSolverInstall, _PreparedAmrFieldSolverInstall])
def installation(request, monkeypatch):
    monkeypatch.setattr(
        "pops.runtime._component_execution_context.component_execution_data",
        lambda context: {"execution_identity": context},
    )
    authorities = [
        {"component_id": name, "component_manifest_identity": name + "-manifest",
         "native_interface": {"name": name}, "parameters": {"z": 2, "a": 1}}
        for name in ("topology", "solver")
    ]
    components = {
        row["component_id"]: NS(
            component_manifest=NS(token=row["component_manifest_identity"]),
            interface=NS(to_data=lambda row=row: dict(row["native_interface"])),
            native_handle=row["component_id"] + "-handle",
        )
        for row in authorities
    }
    contract = {"options": {"relative_tolerance": 1e-8, "absolute_tolerance": 1e-12,
                             "max_iterations": 20}, "schema_identity": "solver-schema"}
    options = {
        "provider_slot": "field-slot", "provider_identity_text": "field-provider",
        "nullspace_provider": {}, "boundary_faces": [], "provider_pack": [],
        "output_route": {"owner_identity": {"owner": "block"}, "owner_block": "block", "key": "phi"},
        "hierarchy_policy": {"policy_id": "hierarchy", "interface_version": 1,
                             "option_schema": "hierarchy-schema", "options": {}},
    }
    data = {"component_bindings": authorities, "native_contract": contract,
            "topology_contract": {"provider_id": "topology-provider", "topology_identity": "mesh"}}
    binding = NS(resolution=NS(to_data=lambda: data, native_contract=contract),
                 facts=NS(layout={"topology_identity": "mesh"}), identity="binding")
    plan = NS(components=components, execution_context="execution",
              artifact=NS(layout_plan=NS(qualified_id="layout")))
    field = NS(name="potential", identity=NS(token="field-id"),
               native_install_data=lambda: options, output_publication_data=lambda: {})
    engine = Mock()
    engine.register_field_solver_provider.return_value = "field-slot"
    return request.param(engine, field, plan), binding, authorities


def test_component_registration_preserves_exact_arguments_and_plan_order(installation):
    adapter, binding, authorities = installation
    adapter.install_component(binding)
    engine = adapter.engine
    arguments = engine.register_field_solver_provider.call_args.args
    assert arguments[:5] == ("field-slot", "topology-handle", "solver-handle", *authorities)
    assert arguments[5:9] == ('{"a":1,"z":2}', '{"a":1,"z":2}', "layout", "mesh")
    import json
    boundary = json.loads(arguments[9])
    assert boundary["faces"] == [] and boundary["nullspace_provider"] == {}
    assert boundary["topology_identity"] == "mesh" and boundary["identity"]
    assert arguments[10:] == (1e-8, 1e-12, 20, {"execution_identity": "execution"})
    calls = [call[0] for call in engine.mock_calls]
    assert calls[:2] == ["register_field_solver_provider", "set_field_solver_plan"]
    if isinstance(adapter, _PreparedAmrFieldSolverInstall):
        assert calls[2:] == ["_set_field_topology_authority"]


@pytest.mark.parametrize("failure, message", [
    ("plan", "authenticated InstallPlan"),
    ("count", "exact topology and solver bindings"),
    ("missing", "requires installed component"),
    ("manifest", "manifest identity changed"),
    ("interface", "native interface identity changed"),
    ("unloaded", "must be loaded"),
])
def test_component_authentication_fails_before_native_side_effects(installation, failure, message):
    adapter, binding, authorities = installation
    component = adapter.install_plan.components["solver"]
    if failure == "plan":
        adapter.install_plan = None
    elif failure == "count":
        authorities.pop()
    elif failure == "missing":
        del adapter.install_plan.components["solver"]
    elif failure == "manifest":
        component.component_manifest.token = "changed"
    elif failure == "interface":
        component.interface.to_data = lambda: {"name": "changed"}
    elif failure == "unloaded":
        component.native_handle = None
    with pytest.raises(ValueError, match=message):
        adapter.install_component(binding)
    assert adapter.engine.mock_calls == []


@pytest.mark.parametrize("invalid_identity", [None, "", 1])
def test_invalid_native_identity_never_publishes_plan(installation, invalid_identity):
    adapter, binding, _ = installation
    adapter.engine.register_field_solver_provider.return_value = invalid_identity
    with pytest.raises(RuntimeError, match="no exact identity"):
        adapter.install_component(binding)
    adapter.engine.set_field_solver_plan.assert_not_called()


def test_component_provider_route_preserves_each_engine_contract(installation):
    adapter, binding, _ = installation
    adapter.engine.register_field_solver_provider.return_value = "different-slot"
    if isinstance(adapter, _PreparedAmrFieldSolverInstall):
        with pytest.raises(RuntimeError, match="changed its provider route"):
            adapter.install_component(binding)
        adapter.engine.set_field_solver_plan.assert_not_called()
    else:
        adapter.install_component(binding)
        assert adapter.engine.set_field_solver_plan.call_args.args[-1] == "field-slot"


def test_registered_nullspace_preserves_its_resolved_native_contract():
    from pops.runtime._field_provider_install import PreparedFieldNullspaceInstall

    engine = Mock()
    contract = {"provider_route": "mean-zero", "schema_identity": "nullspace-schema",
                "options": {"tolerance": 1e-12}}
    binding = NS(resolution=NS(to_data=lambda: {"native_contract": contract}))
    PreparedFieldNullspaceInstall(engine, "field-slot").install_registered_nullspace(binding)
    engine.set_field_nullspace.assert_called_once_with(
        "field-slot", "mean-zero", "nullspace-schema", {"tolerance": 1e-12})
