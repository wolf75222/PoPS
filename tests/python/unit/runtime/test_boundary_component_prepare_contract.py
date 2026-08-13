"""Boundary bindings preserve the authenticated component preparation contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from pops._platform_contracts import ExecutionContext, ExecutionResource, proven_serial_manifest
from pops.runtime._component_execution_context import component_execution_data
from pops.runtime._runtime_authorities import (
    _boundary_face_ordinal,
    _install_boundary_authorities,
    _periodic_identification_rows,
    install_runtime_authorities,
)


ROOT = Path(__file__).resolve().parents[4]


def test_native_boundary_install_has_no_component_count_compatibility_abi():
    old_scalar_adapter = "const std::vector<double>& face_values, int ncomp"
    typed_roles = "const std::vector<std::string>& component_roles"
    for relative in (
        "include/pops/runtime/system.hpp",
        "include/pops/runtime/amr_system.hpp",
        "src/runtime/system/system_install.cpp",
        "src/runtime/amr/amr_system.cpp",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert old_scalar_adapter not in source
        assert typed_roles in source


def test_native_boundary_rollback_uses_typed_staged_package_kind():
    source = (ROOT / "src/runtime/system/system_install.cpp").read_text(encoding="utf-8")
    implementation = (ROOT / "src/runtime/system/system_impl.hpp").read_text(encoding="utf-8")

    assert "enum class NativePackageKind { generic, prepared_boundary };" in (
        ROOT / "include/pops/runtime/system.hpp"
    ).read_text(encoding="utf-8")
    assert "NativePackageKind kind = NativePackageKind::generic;" in implementation
    assert source.count("NativePackageKind::prepared_boundary") >= 4
    assert "NativePackageKind::generic);" in source
    assert "package.kind == NativePackageKind::prepared_boundary" in source
    assert "package.identity.starts_with" not in source


def _execution_context() -> ExecutionContext:
    backend = proven_serial_manifest(
        backend="production", target="system", abi="test|clang++|c++23", runtime=True
    )
    return ExecutionContext(
        backend=backend,
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", "float64"),
        device=ExecutionResource("device", "host"),
    )


@pytest.mark.parametrize(("dimension", "axis"), ((1, 0), (3, 2)))
def test_ranked_periodicity_uses_face_table_without_a_second_identity_row(dimension, axis):
    def endpoint(side):
        return {
            "qualified_id": "case::axis%d::%s" % (axis, side),
            "orientation": {
                "schema_version": 1,
                "axis": axis,
                "side": side,
                "outward_sign": -1 if side == "lower" else 1,
            },
        }

    source_face = 2 * axis
    target_face = source_face + 1
    face_types = ["foextrap"] * (2 * dimension)
    face_types[source_face] = face_types[target_face] = "periodic"
    data = {
        "periodic_identifications": [
            {
                "source": endpoint("lower"),
                "target": endpoint("upper"),
                "source_face": source_face,
                "target_face": target_face,
                "permutation": list(range(dimension)),
                "signs": [1] * dimension,
            }
        ],
    }

    assert (
        _boundary_face_ordinal(endpoint("upper"), dimension=dimension, where="test") == target_face
    )
    assert _periodic_identification_rows(data, face_types, dimension=dimension) == []


def test_mapped_periodicity_preserves_one_dynamic_rank_three_row_for_provider_routing():
    def endpoint(axis, side):
        return {
            "qualified_id": "case::axis%d::%s" % (axis, side),
            "orientation": {
                "schema_version": 1,
                "axis": axis,
                "side": side,
                "outward_sign": -1 if side == "lower" else 1,
            },
        }

    face_types = ["foextrap"] * 6
    face_types[0] = face_types[3] = "periodic"
    data = {
        "periodic_identifications": [
            {
                "source": endpoint(0, "lower"),
                "target": endpoint(1, "upper"),
                "source_face": 0,
                "target_face": 3,
                "permutation": [1, 0, 2],
                "signs": [1, 1, 1],
            }
        ],
    }

    assert _periodic_identification_rows(data, face_types, dimension=3) == [
        [0, 3, 1, 0, 2, 1, 1, 1]
    ]


@pytest.mark.parametrize("prepare_fails", (False, True))
@pytest.mark.parametrize("dimension", (1, 2, 3))
@pytest.mark.parametrize("adaptive", (False, True))
@pytest.mark.parametrize(
    ("operation", "native_interface", "expected_installer"),
    (
        ("apply_region_batch", {"abi_id": 17, "version": 1, "cpp_table": "GhostBoundary"}, "ghost"),
        ("transform_faces", {"abi_id": 6, "version": 1, "cpp_table": "BoundaryFlux"}, "flux"),
    ),
)
def test_boundary_component_install_is_transactional_and_preserves_prepare_json(
    prepare_fails, dimension, adaptive, operation, native_interface, expected_installer
):
    component_id = "pops://external.test/boundary@1.0.0"
    manifest_identity = "component-manifest:boundary-test"
    region = {
        "kind": "face",
        "dimension": dimension,
        "codimension": 1,
        "axes": [0],
        "sides": [-1],
        "identity": "left-face",
    }
    component_row = {
        "target": {"qualified_id": "case::block::left-boundary"},
        "component_id": component_id,
        "component_manifest_identity": manifest_identity,
        "native_interface": native_interface,
        "interface_version": 1,
        "region": region,
        "parameters": [{"qualified_id": "case::inlet", "value": 2.0}],
        "operation": operation,
        "state_identity": "case::block::state",
        "states": [],
        "directions": [],
        "fields": [],
        "outputs": ["case::block::state"],
    }
    runtime_data = {
        "schema_version": 1,
        "authority_type": "prepared_boundary_plan",
        "identity": "case::block::boundary-plan",
        "state": {"qualified_id": "case::block::state"},
        "required_depth": 1,
        "faces": [
            {
                "ordinal": ordinal,
                "producer": "case::block::boundary::face::%d" % ordinal,
                "type": "foextrap",
                "values": [0.0],
            }
            for ordinal in range(2 * dimension)
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
            self.prepare_fails = prepare_fails
            self.state_routes = []
            self.staged_packages = [("generic", "riemann")]
            self.installer = None
            self.events = []

        def _install_block_state_route(self, block, identity):
            self.events.append("state-route")
            self.state_routes.append((block, identity))

        def _install_boundary_plan(self, *args):
            self.events.append("boundary-plan")

        def _discard_boundary_plans(self):
            self.discarded = True
            self.events.append("discard")
            self.state_routes.clear()
            self.staged_packages[:] = [
                package for package in self.staged_packages if package[0] != "prepared_boundary"
            ]

        if adaptive:

            def _prepare_boundary_execution_lane(self, communicator_authority, execution_data):
                assert communicator_authority is None
                assert execution_data == component_execution_data(execution_context)
                self.events.append("boundary-lane")
        else:

            def _prepare_boundary_execution_lane(self, communicator_authority, execution_identity):
                assert communicator_authority is None
                assert execution_identity == execution_context.identity.token
                self.events.append("boundary-lane")

        def _install_ghost_boundary_component(
            self, block, handle, row, parameters_json, target_json, execution
        ):
            self._install_component(
                "ghost", block, handle, row, parameters_json, target_json, execution
            )

        def _preflight_ghost_boundary_component(
            self, handle, row, parameters_json, target_json, execution
        ):
            assert handle is native_handle
            assert row == component_row
            assert parameters_json == target_json == ""
            assert execution["communicator_identity"] == "serial"
            self.events.append("ghost-preflight")

        def _preflight_boundary_flux_component(
            self, handle, row, parameters_json, target_json, execution
        ):
            assert handle is native_handle
            assert row == component_row
            assert parameters_json == target_json == ""
            assert execution["communicator_identity"] == "serial"
            self.events.append("flux-preflight")

        def _install_boundary_flux_component(
            self, block, handle, row, parameters_json, target_json, execution
        ):
            self._install_component(
                "flux", block, handle, row, parameters_json, target_json, execution
            )

        def _install_component(
            self, installer, block, handle, row, parameters_json, target_json, execution
        ):
            assert block == "block"
            assert handle is native_handle
            assert row == component_row
            assert execution["communicator_identity"] == "serial"
            self.events.append("%s-install" % installer)
            self.installer = installer
            self.prepare_overrides = (parameters_json, target_json)
            self.staged_packages.append(("prepared_boundary", installer))
            if self.prepare_fails:
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
    execution_context = _execution_context()
    installed = SimpleNamespace(
        component_manifest=SimpleNamespace(token=manifest_identity),
        interface=Interface(),
        native_handle=native_handle,
    )
    artifact = SimpleNamespace(
        resolved_dimension=dimension,
        blocks=(
            SimpleNamespace(name="block", model=SimpleNamespace(n_vars=1, cons_roles=("Scalar",))),
        ),
        plan=SimpleNamespace(blocks=(BoundaryBlock(),), field_plans={}),
        layout_plan=SimpleNamespace(layouts=(SimpleNamespace(adaptive=adaptive),)),
    )
    install_plan = SimpleNamespace(
        artifact=artifact,
        params={},
        components={component_id: installed},
        execution_context=execution_context,
    )

    if prepare_fails:
        with pytest.raises(RuntimeError, match="component prepare rejected"):
            install_runtime_authorities(engine, install_plan)
        assert native.discarded is True
        assert not hasattr(engine, "_boundary_authorities")
        native.prepare_fails = False
        install_runtime_authorities(engine, install_plan)
        assert native.state_routes == [("block", "case::block::state")]
        assert native.staged_packages == [
            ("generic", "riemann"),
            ("prepared_boundary", expected_installer),
        ]
        assert hasattr(engine, "_boundary_authorities")
    else:
        if adaptive:
            _install_boundary_authorities(engine, install_plan)
        else:
            install_runtime_authorities(engine, install_plan)
        assert native.state_routes == [("block", "case::block::state")]
        assert native.discarded is False
    assert native.prepare_overrides == ("", "")
    assert native.installer == expected_installer
    if operation == "apply_region_batch":
        preflight = "ghost-preflight"
        assert (
            native.events.index(preflight)
            < native.events.index("boundary-lane")
            < native.events.index("state-route")
        )
    else:
        assert "ghost-preflight" not in native.events
        preflight = "flux-preflight"
        assert (
            native.events.index(preflight)
            < native.events.index("boundary-lane")
            < native.events.index("state-route")
        )
    assert native.events.count(preflight) == 1 + prepare_fails
    if prepare_fails:
        preflight_indices = [
            index for index, event in enumerate(native.events) if event == preflight
        ]
        assert preflight_indices[0] < native.events.index("discard") < preflight_indices[1]


@pytest.mark.parametrize(
    ("target_axis", "target_face", "permutation", "signs", "face_types"),
    (
        (0, 1, [0, 1], [1, -1], ["periodic", "periodic", "dirichlet", "foextrap"]),
        (1, 3, [1, 0], [1, 1], ["periodic", "foextrap", "dirichlet", "periodic"]),
    ),
)
def test_signed_periodic_identification_reaches_native_install_without_callback(
    target_axis, target_face, permutation, signs, face_types
):
    def boundary_identity(name, axis, side):
        return {
            "qualified_id": "case::%s" % name,
            "orientation": {
                "schema_version": 1,
                "axis": axis,
                "side": side,
                "outward_sign": -1 if side == "lower" else 1,
            },
        }

    source = boundary_identity("xlo", 0, "lower")
    target = boundary_identity("target", target_axis, "upper")
    runtime_data = {
        "schema_version": 1,
        "authority_type": "prepared_boundary_plan",
        "identity": "case::block::reflected-periodic-plan",
        "state": {"qualified_id": "case::block::state"},
        "required_depth": 2,
        "faces": [
            {
                "ordinal": ordinal,
                "producer": "case::block::reflected-periodic::face::%d" % ordinal,
                "type": face_types[ordinal],
                "representation": "conservative",
                "values": [0.0],
                "analytic_programs": (
                    [{"opcodes": ["x", "input", "add"], "literals": [0.0, 0.0, 0.0]}]
                    if ordinal == 2
                    else []
                ),
                "analytic_clock": "clock.analytic" if ordinal == 2 else None,
            }
            for ordinal in range(4)
        ],
        "omitted_interface_faces": [],
        "periodic_identifications": [
            {
                "source": source,
                "target": target,
                "source_face": 0,
                "target_face": target_face,
                "permutation": permutation,
                "signs": signs,
            }
        ],
        "component_regions": [],
        "interface_component_bindings": [],
        "interface_endpoints": [],
    }

    class Authority:
        def runtime_boundary_data(self, params):
            assert params == {}
            return deepcopy(runtime_data)

    class Native:
        def __init__(self):
            self.installed = None

        def _install_block_state_route(self, block, identity):
            assert (block, identity) == ("block", "case::block::state")

        def _install_boundary_plan(self, *args):
            self.installed = args

        def _prepare_boundary_execution_lane(self, communicator_authority, execution_identity):
            assert communicator_authority is None
            assert execution_identity == execution_context.identity.token

        def _discard_boundary_plans(self):
            raise AssertionError("valid signed periodic installation must not roll back")

    class BoundaryBlock:
        name = "block"
        state_identities = ("case::block::state",)
        boundaries = (Authority(),)

    native = Native()
    engine = SimpleNamespace(_s=native)
    execution_context = _execution_context()
    artifact = SimpleNamespace(
        resolved_dimension=2,
        blocks=(
            SimpleNamespace(name="block", model=SimpleNamespace(n_vars=1, cons_roles=("Scalar",))),
        ),
        plan=SimpleNamespace(blocks=(BoundaryBlock(),), field_plans={}),
        layout_plan=SimpleNamespace(layouts=(SimpleNamespace(adaptive=False),)),
    )
    install_plan = SimpleNamespace(
        artifact=artifact,
        params={},
        components={},
        execution_context=execution_context,
    )

    install_runtime_authorities(engine, install_plan)

    assert native.installed is not None
    assert native.installed[3] == face_types
    assert native.installed[5] == [
        "case::block::reflected-periodic::face::0",
        "case::block::reflected-periodic::face::1",
        "case::block::reflected-periodic::face::2",
        "case::block::reflected-periodic::face::3",
    ]
    assert native.installed[6] == ["Scalar"]
    assert native.installed[9] == [[0, target_face, *permutation, *signs]]
    assert native.installed[10] == ["conservative"] * 4
    assert native.installed[11] == [""] * 4
    assert native.installed[12] == [[], [], ["x", "input", "add"], []]
    assert native.installed[13] == [[], [], [0.0, 0.0, 0.0], []]
    assert native.installed[14] == ["", "", "clock.analytic", ""]
