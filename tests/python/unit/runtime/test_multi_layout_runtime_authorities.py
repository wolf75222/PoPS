"""Multi-layout Uniform children receive per-layout RuntimeInstance authorities."""

from __future__ import annotations

from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from pops._platform_contracts import ExecutionContext, ExecutionResource, proven_serial_manifest
from pops.runtime._multi_layout_executor import (
    _layout_runtime_authority_plan,
    _release_layout_engines,
    install_multi_layout_uniform,
)
from pops.runtime._runtime_authorities import (
    finalize_layout_runtime_authorities,
    install_runtime_authorities,
)
from pops.time import FixedDt


def _execution_context():
    backend = proven_serial_manifest(
        backend="production", target="system", abi="test|clang++|c++23", runtime=True
    )
    return ExecutionContext(
        backend=backend,
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", "float64"),
        device=ExecutionResource("device", "host"),
    )


INTERFACE_ID = "case::shared-flux"
FLUX_COMPONENT_ID = "pops://runtime.test/flux@1.0.0"
FLUX_MANIFEST = "component-manifest:shared-flux"
FLUX_INTERFACE = {"abi_id": 3, "version": 1, "cpp_table": "NumericalFlux"}


def _boundary_data(
    block_name,
    state_identity,
    *,
    dimension=2,
    interface_component_bindings=None,
    interface_endpoints=None,
):
    return {
        "schema_version": 1,
        "authority_type": "prepared_boundary_plan",
        "identity": "%s::boundary-plan" % block_name,
        "state": {"qualified_id": state_identity},
        "required_depth": 1,
        "faces": [
            {
                "ordinal": ordinal,
                "producer": "%s::boundary::face::%d" % (block_name, ordinal),
                "type": "periodic",
                "values": [0.0],
            }
            for ordinal in range(2 * dimension)
        ],
        "omitted_interface_faces": [],
        "component_regions": [],
        "interface_component_bindings": list(interface_component_bindings or ()),
        "interface_endpoints": list(interface_endpoints or ()),
    }


def _shared_interface_body(layout_id):
    return {
        "handle": {"qualified_id": INTERFACE_ID},
        "left": {"layout": {"qualified_id": layout_id}},
        "right": {"layout": {"qualified_id": layout_id}},
    }


def _shared_interface_component():
    return {
        "operation": "evaluate_faces",
        "component_id": FLUX_COMPONENT_ID,
        "component_manifest_identity": FLUX_MANIFEST,
        "native_interface": dict(FLUX_INTERFACE),
        "interface_version": 1,
    }


def _shared_interface_binding(layout_id):
    return {
        "interface": _shared_interface_body(layout_id),
        "component": _shared_interface_component(),
    }


class _Authority:
    def __init__(
        self,
        block_name,
        state_identity,
        *,
        interface_component_bindings=None,
        interface_endpoints=None,
    ):
        self._block_name = block_name
        self._state_identity = state_identity
        self._interface_component_bindings = list(interface_component_bindings or ())
        self._interface_endpoints = list(interface_endpoints or ())

    def runtime_boundary_data(self, params):
        assert params == {}
        return deepcopy(
            _boundary_data(
                self._block_name,
                self._state_identity,
                interface_component_bindings=self._interface_component_bindings,
                interface_endpoints=self._interface_endpoints,
            )
        )


class _PlanBlock:
    def __init__(
        self,
        name,
        state_identity,
        *,
        interface_component_bindings=None,
        interface_endpoints=None,
    ):
        self.name = name
        self.state_identities = (state_identity,)
        self.boundaries = (
            _Authority(
                name,
                state_identity,
                interface_component_bindings=interface_component_bindings,
                interface_endpoints=interface_endpoints,
            ),
        )


def _two_layout_install_plan(execution_context):
    fine_layout = SimpleNamespace(qualified_id="layout::fine", adaptive=False)
    coarse_layout = SimpleNamespace(qualified_id="layout::coarse", adaptive=False)
    compiled_blocks = (
        SimpleNamespace(name="tracer", model=SimpleNamespace(n_vars=1, cons_roles=("Scalar",))),
        SimpleNamespace(name="coarse", model=SimpleNamespace(n_vars=1, cons_roles=("Scalar",))),
    )
    plan_blocks = (
        _PlanBlock("tracer", "case::tracer::state"),
        _PlanBlock("coarse", "case::coarse::state"),
    )
    layout_plan = SimpleNamespace(
        layouts=(fine_layout, coarse_layout),
        assignments=(
            SimpleNamespace(
                subject_kind="block",
                subject=SimpleNamespace(local_id="tracer"),
                layout=fine_layout,
            ),
            SimpleNamespace(
                subject_kind="block",
                subject=SimpleNamespace(local_id="coarse"),
                layout=coarse_layout,
            ),
        ),
        mappings=(),
    )
    return SimpleNamespace(
        artifact=SimpleNamespace(
            resolved_dimension=2,
            blocks=compiled_blocks,
            plan=SimpleNamespace(blocks=plan_blocks, field_plans={}, capabilities={}),
            layout_plan=layout_plan,
            layout_programs=(
                SimpleNamespace(layout_id="layout::fine", block_names=("tracer",)),
                SimpleNamespace(layout_id="layout::coarse", block_names=("coarse",)),
            ),
        ),
        params={},
        components={},
        execution_context=execution_context,
        instances={"tracer": object(), "coarse": object()},
        aux={},
    )


def test_layout_runtime_authority_plan_exposes_only_selected_blocks():
    plan = _two_layout_install_plan(_execution_context())

    projected = _layout_runtime_authority_plan(plan, ("tracer",))

    assert tuple(row.name for row in projected.artifact.blocks) == ("tracer",)
    assert tuple(row.name for row in projected.artifact.plan.blocks) == ("tracer",)
    assert projected.artifact.plan.blocks[0].state_identities == ("case::tracer::state",)
    assert projected.artifact.resolved_dimension == 2
    assert projected.artifact.plan.field_plans is plan.artifact.plan.field_plans
    assert projected.artifact.plan.capabilities == {}
    assert projected.artifact.layout_plan is plan.artifact.layout_plan
    assert projected.params is plan.params
    assert projected.components is plan.components
    assert projected.execution_context is plan.execution_context
    with pytest.raises(ValueError, match="unique selected block names"):
        _layout_runtime_authority_plan(plan, ())
    with pytest.raises(ValueError, match="unique selected block names"):
        _layout_runtime_authority_plan(plan, ("tracer", "tracer"))
    with pytest.raises(ValueError, match="selected compiled/plan block set"):
        _layout_runtime_authority_plan(plan, ("missing",))


def test_install_runtime_authorities_prepares_exact_lane_then_selected_state_routes():
    execution_context = _execution_context()
    plan = _two_layout_install_plan(execution_context)
    projected = _layout_runtime_authority_plan(plan, ("tracer",))

    class Native:
        def __init__(self):
            self.events = []
            self.state_routes = []
            self.lane_arguments = None

        def _install_block_state_route(self, block, identity):
            self.events.append("state-route")
            self.state_routes.append((block, identity))

        def _install_boundary_plan(self, *args):
            self.events.append("boundary-plan")

        def _prepare_boundary_execution_lane(self, communicator_authority, execution_identity):
            assert communicator_authority is execution_context.communicator.handle
            assert execution_identity == execution_context.identity.token
            assert execution_identity not in {None, "WORLD", "MPI_COMM_WORLD"}
            self.lane_arguments = (communicator_authority, execution_identity)
            self.events.append("boundary-lane")

        def _discard_boundary_plans(self):
            raise AssertionError("valid per-layout authority install must not roll back")

    native = Native()
    engine = SimpleNamespace(_s=native)

    install_runtime_authorities(engine, projected)

    assert native.lane_arguments == (
        execution_context.communicator.handle,
        execution_context.identity.token,
    )
    assert native.state_routes == [("tracer", "case::tracer::state")]
    assert "coarse" not in {block for block, _identity in native.state_routes}
    assert native.events.index("boundary-lane") < native.events.index("state-route")
    assert "compiled" not in native.events


def test_two_layout_projection_preserves_independent_selected_sets():
    plan = _two_layout_install_plan(_execution_context())

    fine = _layout_runtime_authority_plan(plan, ("tracer",))
    coarse = _layout_runtime_authority_plan(plan, ("coarse",))

    assert {row.name for row in fine.artifact.blocks} == {"tracer"}
    assert {row.name for row in coarse.artifact.blocks} == {"coarse"}
    assert {row.name for row in fine.artifact.plan.blocks} == {"tracer"}
    assert {row.name for row in coarse.artifact.plan.blocks} == {"coarse"}
    assert fine.artifact.layout_plan is plan.artifact.layout_plan
    assert coarse.artifact.layout_plan is plan.artifact.layout_plan
    assert {row.qualified_id for row in plan.artifact.layout_plan.layouts} == {
        "layout::fine",
        "layout::coarse",
    }
    assert {tuple(row.block_names) for row in plan.artifact.layout_programs} == {
        ("tracer",),
        ("coarse",),
    }


def test_release_layout_engines_destroys_children_in_reverse_order():
    released = []

    class Child:
        def __init__(self, name):
            self.name = name
            self._s = SimpleNamespace(name=name)

        def destroy(self):
            released.append(self.name)
            self._s = None

    children = [Child("fine"), Child("coarse")]
    _release_layout_engines(children)
    assert released == ["coarse", "fine"]
    assert children == []


def test_later_child_failure_releases_earlier_children_in_reverse_order(monkeypatch):
    execution_context = _execution_context()
    plan = _two_layout_install_plan(execution_context)
    strategy = FixedDt(1.0e-3)
    released = []
    child_events = {}

    class FakeLayouts:
        def __init__(self):
            fine = SimpleNamespace(qualified_id="layout::fine")
            coarse = SimpleNamespace(qualified_id="layout::coarse")
            self.rows = (
                SimpleNamespace(handle=fine),
                SimpleNamespace(handle=coarse),
            )
            self.plan = SimpleNamespace(
                layouts=self.rows,
                normalized=lambda handle: SimpleNamespace(
                    native_spatial_layout=handle.qualified_id,
                    transition_ratios=(),
                ),
            )

    class FakeSystem:
        def __init__(self, config):
            self.config = config
            self.events = []
            self._s = SimpleNamespace(layout=config)
            child_events[config] = self.events

        def destroy(self):
            released.append(self.config)
            self._s = None

        def _install_compiled(self, *args, **kwargs):
            self.authority_plan = kwargs.get("authority_plan")
            self.events.append("compiled")

    def install_authorities(engine, authority_plan):
        engine.events.append("authorities")
        engine._boundary_authorities = {}
        names = tuple(row.name for row in authority_plan.artifact.blocks)
        if names == ("coarse",):
            raise RuntimeError("injected later-child authority failure")

    plan.layout = FakeLayouts()
    plan.bind_identity = object()
    plan.artifact.bind_schema = object()
    plan.artifact.semantic_identity = object()
    plan.artifact.artifact_identity = object()
    plan.artifact.native_layouts = {
        "layout::fine": "fine-native",
        "layout::coarse": "coarse-native",
    }

    def _layout_program(layout_id, block_name):
        return SimpleNamespace(
            layout_id=layout_id,
            block_names=(block_name,),
            target="system",
            program=SimpleNamespace(
                so_path="%s.so" % layout_id,
                program=SimpleNamespace(
                    _step_strategy=strategy,
                    transaction_plan=lambda: "shared-transaction",
                ),
            ),
        )

    plan.artifact.layout_programs = (
        _layout_program("layout::fine", "tracer"),
        _layout_program("layout::coarse", "coarse"),
    )
    runtime_plan = SimpleNamespace(communication=SimpleNamespace(transfers=()))

    fake_system = ModuleType("pops.runtime._system")
    fake_system.System = FakeSystem
    monkeypatch.setitem(sys.modules, "pops.runtime._system", fake_system)
    monkeypatch.setattr(
        "pops.runtime._multi_layout_executor._require_runtime_plan_bundle",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "pops.codegen._layout_resolution.ResolvedRuntimeLayouts",
        FakeLayouts,
    )
    monkeypatch.setattr(
        "pops.runtime._runtime_mesh_lowering.system_config_from_layout",
        lambda layout: layout,
    )
    monkeypatch.setattr(
        "pops.runtime._runtime_mesh_lowering.install_embedded_boundary",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "pops.runtime._checkpoint_spatial.install_checkpoint_spatial_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "pops.runtime._runtime_executor._uniform_initial_sources",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        "pops.runtime._runtime_authorities.install_runtime_authorities",
        install_authorities,
    )

    with pytest.raises(RuntimeError, match="injected later-child authority failure"):
        install_multi_layout_uniform(plan, runtime_plan)

    assert released == ["coarse-native", "fine-native"]
    assert child_events["fine-native"] == ["authorities", "compiled"]
    assert child_events["coarse-native"] == ["authorities"]


class _AuthorityNative:
    def __init__(self, *, block_names=(), install_provider=None):
        self.events = []
        self.state_routes = []
        self.jobs = None
        self._block_names = tuple(block_names)
        self._install_provider = install_provider

    def _install_block_state_route(self, block, identity):
        self.events.append("state-route")
        self.state_routes.append((block, identity))

    def _install_boundary_plan(self, *args):
        self.events.append("boundary-plan")

    def _prepare_boundary_execution_lane(self, communicator_authority, execution_identity):
        assert execution_identity not in {None, "WORLD", "MPI_COMM_WORLD"}
        self.events.append("boundary-lane")

    def _discard_boundary_plans(self):
        raise AssertionError("valid per-layout authority install must not roll back")

    def block_names(self):
        return self._block_names

    def _install_interface_flux_provider(self, jobs):
        self.events.append("interface-provider")
        self.jobs = list(jobs)
        if self._install_provider is not None:
            self._install_provider(jobs)


def _flux_component():
    class Interface:
        version = 1

        @staticmethod
        def to_data():
            return dict(FLUX_INTERFACE)

    return SimpleNamespace(
        component_manifest=SimpleNamespace(token=FLUX_MANIFEST),
        interface=Interface(),
        native_handle=object(),
    )


def test_unfinalized_shared_interface_declarations_fail_closed():
    left = _PlanBlock(
        "fluid",
        "case::fluid::state",
        interface_component_bindings=(_shared_interface_binding("layout::pair"),),
        interface_endpoints=({"interface": INTERFACE_ID, "owned_sides": ["left"]},),
    )
    engine = SimpleNamespace(
        _s=_AuthorityNative(block_names=("fluid",)),
        _boundary_authorities={"fluid": left.boundaries[0].runtime_boundary_data({})},
    )

    with pytest.raises(RuntimeError, match="unfinalized shared-interface"):
        finalize_layout_runtime_authorities(engine, None)
    assert getattr(engine, "_interface_authorities", None) is None


def test_one_sided_layout_projection_refuses_unfinalized_shared_interface():
    execution_context = _execution_context()
    binding = _shared_interface_binding("layout::fine")
    plan = _two_layout_install_plan(execution_context)
    plan.artifact.plan.blocks = (
        _PlanBlock(
            "tracer",
            "case::tracer::state",
            interface_component_bindings=(binding,),
            interface_endpoints=({"interface": INTERFACE_ID, "owned_sides": ["left"]},),
        ),
        _PlanBlock(
            "coarse",
            "case::coarse::state",
            interface_component_bindings=(binding,),
            interface_endpoints=({"interface": INTERFACE_ID, "owned_sides": ["right"]},),
        ),
    )
    plan.components = {FLUX_COMPONENT_ID: _flux_component()}
    projected = _layout_runtime_authority_plan(plan, ("tracer",))
    native = _AuthorityNative(block_names=("tracer",))
    engine = SimpleNamespace(_s=native)

    install_runtime_authorities(engine, projected)
    assert engine._boundary_authorities["tracer"]["interface_component_bindings"]
    assert engine._boundary_authorities["tracer"]["interface_endpoints"]

    with pytest.raises(ValueError, match="BoundaryHandle must identify exactly one native block"):
        finalize_layout_runtime_authorities(engine, projected)
    assert getattr(engine, "_interface_authorities", None) is None


def test_same_layout_shared_interface_finalizes_through_native_provider():
    execution_context = _execution_context()
    binding = _shared_interface_binding("layout::pair")
    left = _PlanBlock(
        "fluid",
        "case::fluid::state",
        interface_component_bindings=(binding,),
        interface_endpoints=({"interface": INTERFACE_ID, "owned_sides": ["left"]},),
    )
    right = _PlanBlock(
        "wall",
        "case::wall::state",
        interface_component_bindings=(binding,),
        interface_endpoints=({"interface": INTERFACE_ID, "owned_sides": ["right"]},),
    )
    layout = SimpleNamespace(qualified_id="layout::pair", adaptive=False)
    plan = SimpleNamespace(
        artifact=SimpleNamespace(
            resolved_dimension=2,
            blocks=(
                SimpleNamespace(
                    name="fluid", model=SimpleNamespace(n_vars=1, cons_roles=("Scalar",))
                ),
                SimpleNamespace(
                    name="wall", model=SimpleNamespace(n_vars=1, cons_roles=("Scalar",))
                ),
            ),
            plan=SimpleNamespace(blocks=(left, right), field_plans={}, capabilities={}),
            layout_plan=SimpleNamespace(
                layouts=(layout,),
                assignments=(
                    SimpleNamespace(
                        subject_kind="block",
                        subject=SimpleNamespace(local_id="fluid"),
                        layout=layout,
                    ),
                    SimpleNamespace(
                        subject_kind="block",
                        subject=SimpleNamespace(local_id="wall"),
                        layout=layout,
                    ),
                ),
            ),
        ),
        params={},
        components={FLUX_COMPONENT_ID: _flux_component()},
        execution_context=execution_context,
    )
    projected = _layout_runtime_authority_plan(plan, ("fluid", "wall"))
    native = _AuthorityNative(block_names=("fluid", "wall"))
    engine = SimpleNamespace(_s=native)

    install_runtime_authorities(engine, projected)
    finalize_layout_runtime_authorities(engine, projected)

    assert native.jobs is not None
    assert len(native.jobs) == 1
    assert native.jobs[0]["left_block"] == 0
    assert native.jobs[0]["right_block"] == 1
    assert engine._interface_authorities[INTERFACE_ID]["left_block"] == "fluid"
    assert engine._interface_authorities[INTERFACE_ID]["right_block"] == "wall"
    assert "interface-provider" in native.events


def _layout_config(name, cells):
    return SimpleNamespace(
        name=name,
        shape=(cells, cells),
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
        coordinate_system="cartesian",
        periodicity=(True, True),
    )


def _mapping_row(mapping_id, source_block, target_block):
    return SimpleNamespace(
        requirement=SimpleNamespace(
            qualified_id=mapping_id,
            source_port=SimpleNamespace(
                subject=SimpleNamespace(block_ref=SimpleNamespace(local_id=source_block))
            ),
            target_port=SimpleNamespace(
                subject=SimpleNamespace(block_ref=SimpleNamespace(local_id=target_block))
            ),
        )
    )


def _transfer_row(mapping_id, source_layout, target_layout):
    return SimpleNamespace(
        mapping_id=mapping_id,
        provider_id="pops://mapping/%s" % mapping_id,
        component_id="cell_average",
        source_layout_id=source_layout,
        target_layout_id=target_layout,
        source_representation_uri="cell-average",
        target_representation_uri="cell-average",
        synchronization_uri="pops://synchronization/before-step@1",
        operation_abi=1,
    )


def test_failed_transfer_prepare_releases_retained_handles_before_children(monkeypatch):
    execution_context = _execution_context()
    plan = _two_layout_install_plan(execution_context)
    strategy = FixedDt(1.0e-3)
    events = []
    fine_cfg = _layout_config("fine", 16)
    coarse_cfg = _layout_config("coarse", 8)

    class RetainingSession:
        def __init__(self, source, target):
            self.source = source
            self.target = target
            source.retained_by.append(self)
            target.retained_by.append(self)
            events.append(("retain", source.owner, target.owner))

        def close(self):
            events.append(("release", self.source.owner, self.target.owner))
            self.source.retained_by.remove(self)
            self.target.retained_by.remove(self)
            self.source = None
            self.target = None

    class Native:
        def __init__(self, owner):
            self.owner = owner
            self.retained_by = []

        def _prepare_layout_transfer(self, target, handle, spec, execution):
            if spec["mapping_identity"] == "map-second":
                raise RuntimeError("injected transfer preparation failure")
            return RetainingSession(self, target)

    class FakeSystem:
        def __init__(self, config):
            self.config = config
            self._s = Native(config.name)

        def destroy(self):
            events.append(("destroy", self.config.name))
            assert self._s.retained_by == [], (
                "transfer session still retained native handle %s" % self.config.name
            )
            self._s = None

        def _install_compiled(self, *args, **kwargs):
            self.authority_plan = kwargs.get("authority_plan")

        def _native_step_target(self):
            return self._s

        def spatial_shape(self):
            return self.config.shape

        def n_vars(self, _block):
            return 1

    class FakeLayouts:
        def __init__(self):
            fine = SimpleNamespace(qualified_id="layout::fine")
            coarse = SimpleNamespace(qualified_id="layout::coarse")
            self.rows = (SimpleNamespace(handle=fine), SimpleNamespace(handle=coarse))
            self.plan = SimpleNamespace(
                layouts=self.rows,
                normalized=lambda handle: SimpleNamespace(
                    native_spatial_layout=handle.qualified_id,
                    transition_ratios=(),
                ),
            )

    plan.layout = FakeLayouts()
    plan.bind_identity = object()
    plan.artifact.bind_schema = object()
    plan.artifact.semantic_identity = object()
    plan.artifact.artifact_identity = object()
    plan.artifact.native_layouts = {"layout::fine": fine_cfg, "layout::coarse": coarse_cfg}
    plan.artifact.layout_plan.mappings = (
        _mapping_row("map-first", "tracer", "coarse"),
        _mapping_row("map-second", "tracer", "coarse"),
    )
    plan.components = {
        "cell_average": SimpleNamespace(
            native_handle=object(),
            component_manifest=SimpleNamespace(token="transfer-manifest"),
        )
    }

    def _layout_program(layout_id, block_name):
        return SimpleNamespace(
            layout_id=layout_id,
            block_names=(block_name,),
            target="system",
            program=SimpleNamespace(
                so_path="%s.so" % layout_id,
                program=SimpleNamespace(
                    _step_strategy=strategy,
                    transaction_plan=lambda: "shared-transaction",
                ),
            ),
        )

    plan.artifact.layout_programs = (
        _layout_program("layout::fine", "tracer"),
        _layout_program("layout::coarse", "coarse"),
    )
    runtime_plan = SimpleNamespace(
        communication=SimpleNamespace(
            transfers=(
                _transfer_row("map-first", "layout::fine", "layout::coarse"),
                _transfer_row("map-second", "layout::fine", "layout::coarse"),
            )
        )
    )

    fake_system = ModuleType("pops.runtime._system")
    fake_system.System = FakeSystem
    monkeypatch.setitem(sys.modules, "pops.runtime._system", fake_system)
    monkeypatch.setattr(
        "pops.runtime._multi_layout_executor._require_runtime_plan_bundle",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "pops.codegen._layout_resolution.ResolvedRuntimeLayouts",
        FakeLayouts,
    )
    monkeypatch.setattr(
        "pops.runtime._runtime_mesh_lowering.system_config_from_layout",
        lambda layout: layout,
    )
    monkeypatch.setattr(
        "pops.runtime._runtime_mesh_lowering.install_embedded_boundary",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "pops.runtime._checkpoint_spatial.install_checkpoint_spatial_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "pops.runtime._runtime_executor._uniform_initial_sources",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        "pops.runtime._runtime_authorities.install_runtime_authorities",
        lambda engine, _plan: setattr(engine, "_boundary_authorities", {}),
    )

    with pytest.raises(RuntimeError, match="injected transfer preparation failure"):
        install_multi_layout_uniform(plan, runtime_plan)

    assert events == [
        ("retain", "fine", "coarse"),
        ("release", "fine", "coarse"),
        ("destroy", "coarse"),
        ("destroy", "fine"),
    ]

