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
from pops.runtime._runtime_authorities import install_runtime_authorities
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


def _boundary_data(block_name, state_identity, *, dimension=2):
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
        "interface_component_bindings": [],
        "interface_endpoints": [],
    }


class _Authority:
    def __init__(self, block_name, state_identity):
        self._block_name = block_name
        self._state_identity = state_identity

    def runtime_boundary_data(self, params):
        assert params == {}
        return deepcopy(_boundary_data(self._block_name, self._state_identity))


class _PlanBlock:
    def __init__(self, name, state_identity):
        self.name = name
        self.state_identities = (state_identity,)
        self.boundaries = (_Authority(name, state_identity),)


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
            plan=SimpleNamespace(blocks=plan_blocks, field_plans={}),
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
            self.events.append("compiled")

    def install_authorities(engine, authority_plan):
        engine.events.append("authorities")
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
