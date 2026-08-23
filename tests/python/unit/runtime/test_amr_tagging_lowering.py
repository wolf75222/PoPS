from __future__ import annotations

from types import SimpleNamespace

import pytest

from pops.model import Handle, OwnerPath
from pops.numerics.indicator_stencils import SECOND_ORDER_AXIS, gradient_stencil
from pops.runtime._runtime_mesh_lowering import flow_bootstrap_tagging
from pops.time import Clock


class _CaptureTagging:
    def __init__(self) -> None:
        self.arguments = None

    def _set_bootstrap_tagging(self, *arguments):
        self.arguments = arguments


def test_field_gradient_uses_qualified_prepared_output_route() -> None:
    field = Handle("potential", kind="field", owner=OwnerPath.case("main"))
    identity = field.canonical_identity()
    stencil = gradient_stencil(SECOND_ORDER_AXIS, dimension=2).to_data()

    class _Tagging:
        qualified_id = "test::resolved-tagging"

        @staticmethod
        def runtime_tagging_data(_params):
            return {
                "schema_version": 1,
                "graph_type": "amr_tagging_runtime",
                "lowerings": [{
                    "schema_version": 1,
                    "node_type": "gradient_above",
                    "lowering": {
                        "kind": "tag_lowering",
                        "local_id": "gradient_above",
                        "qualified_id": "test::gradient-above-lowering",
                    },
                }],
                "refine": {
                    "schema_version": 1,
                    "node_type": "gradient_above",
                    "indicator": identity,
                    "variable": "phi",
                    "threshold": 0.25,
                    "discrete_context": {"stencil_lowering": stencil},
                },
                "coarsen": None,
                "hysteresis": {
                    "hysteresis_type": "min_cycles",
                    "min_cycles": 0,
                    "equality": "hold",
                },
                "conflict_policy": "error",
            }

    plan = SimpleNamespace(
        operator=SimpleNamespace(unknown=field),
        native_options={"output_route": {"components": ("phi",)}},
    )
    native = _CaptureTagging()
    flow_bootstrap_tagging(
        native,
        SimpleNamespace(tagging=_Tagging()),
        {},
        clock_identity="test::clock",
        field_plans={"electrostatic": plan},
    )

    assert native.arguments is not None
    assert native.arguments[:5] == (
        ["field"],
        [field.qualified_id],
        [""],
        ["phi"],
        [0],
    )


def test_field_gradient_refuses_ambiguous_solved_field_identity() -> None:
    field = Handle("potential", kind="field", owner=OwnerPath.case("main"))
    plan = SimpleNamespace(
        operator=SimpleNamespace(unknown=field),
        native_options={"output_route": {"components": ("phi",)}},
    )
    other_plan = SimpleNamespace(
        operator=SimpleNamespace(unknown=field),
        native_options={"output_route": {"components": ("phi",)}},
    )
    bootstrap = SimpleNamespace(
        tagging=SimpleNamespace(
            qualified_id="test::resolved-tagging",
            runtime_tagging_data=lambda _params: {
                "schema_version": 1,
                "graph_type": "amr_tagging_runtime",
            },
        ),
    )

    with pytest.raises(
        ValueError, match="multiple resolved field plans claim solved-field identity"
    ):
        flow_bootstrap_tagging(
            _CaptureTagging(),
            bootstrap,
            {},
            clock_identity="test::clock",
            field_plans={"electrostatic": plan, "screened": other_plan},
        )


def test_prescribed_window_lowers_the_honest_layout_periodicity_trajectory() -> None:
    clock = Clock("prescribed-window")
    window = {
        "schema_version": 1,
        "node_type": "prescribed_window",
        "frame": {"canonical_id": "test::frame"},
        "dimension": 1,
        "clock": clock.to_data(),
        "center": [0.25],
        "half_width": [0.1],
        "velocity": [1.0],
        "trajectory": "constant_velocity_layout_periodicity",
    }
    bootstrap = SimpleNamespace(
        tagging=SimpleNamespace(
            qualified_id="test::prescribed-window",
            runtime_tagging_data=lambda _params: {
                "schema_version": 1,
                "graph_type": "amr_tagging_runtime",
                "lowerings": [{
                    "schema_version": 1,
                    "node_type": "prescribed_window",
                    "lowering": {
                        "kind": "tag_lowering",
                        "local_id": "prescribed_window",
                        "qualified_id": "test::prescribed-window-lowering",
                    },
                }],
                "refine": window,
                "coarsen": None,
                "hysteresis": {"hysteresis_type": "min_cycles", "min_cycles": 0, "equality": "hold"},
                "conflict_policy": "error",
            },
        )
    )
    native = _CaptureTagging()
    flow_bootstrap_tagging(native, bootstrap, {}, clock_identity=clock.qualified_id)

    assert native.arguments is not None
    assert native.arguments[0] == ["geometry"]
    assert native.arguments[5] == [6]
    assert native.arguments[8] == [[0.25, 0.1, 1.0]]
