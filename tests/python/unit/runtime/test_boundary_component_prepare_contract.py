"""Boundary bindings preserve the authenticated component preparation contract."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from pops._platform_contracts import ExecutionContext, ExecutionResource, proven_serial_manifest
from pops.runtime._runtime_authorities import install_runtime_authorities


def _execution_context() -> ExecutionContext:
    backend = proven_serial_manifest(
        backend="production", target="system", abi="test|clang++|c++23", runtime=True)
    return ExecutionContext(
        backend=backend,
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", "float64"),
        device=ExecutionResource("device", "host"),
    )


@pytest.mark.parametrize("prepare_fails", (False, True))
def test_boundary_component_install_is_transactional_and_preserves_prepare_json(prepare_fails):
    component_id = "pops://external.test/boundary@1.0.0"
    manifest_identity = "component-manifest:boundary-test"
    native_interface = {"abi_id": 17, "version": 1, "cpp_table": "GhostBoundary"}
    region = {
        "kind": "face", "dimension": 2, "codimension": 1,
        "axes": [0], "sides": [-1], "identity": "left-face",
    }
    component_row = {
        "target": {"qualified_id": "case::block::left-boundary"},
        "component_id": component_id,
        "component_manifest_identity": manifest_identity,
        "native_interface": native_interface,
        "interface_version": 1,
        "region": region,
        "parameters": [{"qualified_id": "case::inlet", "value": 2.0}],
        "operation": "apply_region_batch",
        "state_identity": "case::block::state",
        "states": [], "directions": [], "fields": [],
        "outputs": ["case::block::state"],
    }
    runtime_data = {
        "schema_version": 1,
        "authority_type": "prepared_boundary_plan",
        "identity": "case::block::boundary-plan",
        "state": {"qualified_id": "case::block::state"},
        "required_depth": 1,
        "faces": [
            {"ordinal": ordinal, "type": "foextrap", "values": [0.0]}
            for ordinal in range(4)
        ],
        "omitted_interface_faces": [],
        "component_regions": [component_row],
        "interface_component_bindings": [],
        "interface_endpoints": [],
    }

    class Authority:
        def runtime_boundary_data(self, params):
            assert params == {}
            return deepcopy(runtime_data)

    class Native:
        def __init__(self):
            self.prepare_overrides = None
            self.discarded = False
            self.state_routes = []

        def _install_block_state_route(self, block, identity):
            self.state_routes.append((block, identity))

        def _install_boundary_plan(self, *args):
            pass

        def _discard_boundary_plans(self):
            self.discarded = True

        def _install_ghost_boundary_component(
                self, block, handle, row, parameters_json, target_json, execution):
            assert block == "block"
            assert handle is native_handle
            assert row == component_row
            assert execution["communicator_identity"] == "serial"
            self.prepare_overrides = (parameters_json, target_json)
            if prepare_fails:
                raise RuntimeError("component prepare rejected")

    class Interface:
        version = 1

        @staticmethod
        def to_data():
            return native_interface

    class BoundaryBlock:
        name = "block"
        state_identities = ("case::block::state",)
        boundaries = (Authority(),)

    native_handle = object()
    native = Native()
    engine = SimpleNamespace(_s=native)
    installed = SimpleNamespace(
        component_manifest=SimpleNamespace(token=manifest_identity),
        interface=Interface(), native_handle=native_handle,
    )
    artifact = SimpleNamespace(
        blocks=(SimpleNamespace(name="block", model=SimpleNamespace(n_vars=1)),),
        plan=SimpleNamespace(blocks=(BoundaryBlock(),), field_plans={}),
        layout_plan=SimpleNamespace(layouts=(SimpleNamespace(adaptive=False),)),
    )
    install_plan = SimpleNamespace(
        artifact=artifact, params={}, components={component_id: installed},
        execution_context=_execution_context(),
    )

    if prepare_fails:
        with pytest.raises(RuntimeError, match="component prepare rejected"):
            install_runtime_authorities(engine, install_plan)
        assert native.discarded is True
        assert not hasattr(engine, "_boundary_authorities")
    else:
        install_runtime_authorities(engine, install_plan)
        assert native.state_routes == [("block", "case::block::state")]
        assert native.discarded is False
    assert native.prepare_overrides == ("", "")
