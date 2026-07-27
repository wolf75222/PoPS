from __future__ import annotations

from types import SimpleNamespace

from pops.model import Handle, OwnerPath
from pops.numerics.indicator_stencils import SECOND_ORDER_AXIS, gradient_stencil
from pops.runtime._runtime_mesh_lowering import flow_bootstrap_tagging


class _CaptureTagging:
    def __init__(self) -> None:
        self.arguments = None

    def _set_bootstrap_tagging(self, *arguments):
        self.arguments = arguments


def test_field_gradient_uses_qualified_prepared_output_route() -> None:
    field = Handle("phi", kind="field", owner=OwnerPath.case("main"))
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
        field_plans={"phi": plan},
    )

    assert native.arguments is not None
    assert native.arguments[:5] == (
        ["field"],
        [field.qualified_id],
        [""],
        ["phi"],
        [0],
    )
