"""Runtime scientific-output routing for exact Uniform and AMR EB sidecars."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from pops.runtime._runtime_consumers import RuntimeOutputSnapshot


ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / "python" / "pops" / "runtime" / "_runtime_consumers.py"


@pytest.mark.parametrize("adaptive", [False, True])
def test_uniform_and_amr_layouts_resolve_the_same_typed_eb_sidecar_route(adaptive: bool) -> None:
    class NativeEngine:
        def output_embedded_boundary_local_pieces(self, name: str, level: int):
            return name, level

    native = NativeEngine()
    owner = SimpleNamespace(
        _executor_for_layout=lambda _layout_id: SimpleNamespace(_s=native),
    )
    layout = SimpleNamespace(
        adaptive=adaptive,
        handle=SimpleNamespace(qualified_id="layout::mesh"),
        options={"embedded_boundary": SimpleNamespace(identity="eb::analytic")},
    )
    geometry = SimpleNamespace(key=("layout::mesh", 1), level=1)

    entry = RuntimeOutputSnapshot(owner)._embedded_boundary_output_entry(layout, geometry)

    assert entry == {
        "geometry": geometry,
        "native_engine": native,
        "method_name": "output_embedded_boundary_local_pieces",
    }


def test_scientific_output_build_routes_both_field_and_diagnostic_eb_sidecars() -> None:
    source = RUNTIME.read_text(encoding="utf-8")

    assert "scientific output has no prepared AMR embedded-boundary sidecar provider" not in source
    assert source.count("self._embedded_boundary_output_entry(layout, geometry)") == 2


def test_eb_sidecar_route_refuses_an_uninstalled_native_provider() -> None:
    owner = SimpleNamespace(
        _executor_for_layout=lambda _layout_id: SimpleNamespace(_s=SimpleNamespace()),
    )
    layout = SimpleNamespace(
        adaptive=True,
        handle=SimpleNamespace(qualified_id="layout::mesh"),
        options={"embedded_boundary": SimpleNamespace(identity="eb::analytic")},
    )

    with pytest.raises(RuntimeError, match="output_embedded_boundary_local_pieces"):
        RuntimeOutputSnapshot(owner)._embedded_boundary_output_entry(
            layout, SimpleNamespace(key=("layout::mesh", 0), level=0)
        )
