"""Source-only authoring/resolve tests; these do not execute a native ALE provider."""
import json

import pytest
import pops
from pops.analytic import coordinates, sin, time
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, GeometryEvolution, MovingControlVolumes, normalize_layout_plan
from pops.codegen._native_spatial_layout import NativeSpatialLayoutError, native_spatial_layouts
from pops.codegen._layout_resolution import LayoutCapabilityError, resolve_native_spatial_layouts
from pops._frozen_data import thaw_data


def request(cells=16, amplitude=.03):
    frame = CartesianDomain("ale-reference", (0.,), (1.,)).frame(Cartesian1D())
    case = pops.Case("ale-contract")
    program = pops.Program("geometry-clock")
    x, = coordinates(frame)
    t = time(program.clock)
    evolution = GeometryEvolution((x + amplitude * sin(6.283185307179586*x) * sin(t),))
    layout = MovingControlVolumes(Uniform(CartesianGrid(frame=frame, cells=(cells,))),
                                 evolution=evolution)
    return case, layout


def test_common_expression_map_is_preserved_and_parameter_changes_identity():
    case, layout = request()
    plan = normalize_layout_plan(layout, owner=case.owner_path.canonical())
    payload = thaw_data(plan.layouts[0].requirements["geometry_evolution"])
    assert payload["schema_version"] == 1
    assert payload["coordinate_map"][0] == layout.evolution.coordinate_map[0].to_data()
    _, changed = request(amplitude=.04)
    assert payload != changed.evolution.to_data()
    assert json.loads(json.dumps(layout.evolution.to_data())) == layout.evolution.to_data()


@pytest.mark.parametrize("cells", [8, 32])
def test_resolve_refuses_dynamic_obligation_before_native_allocation(cells):
    case, layout = request(cells)
    plan = normalize_layout_plan(layout, owner=case.owner_path.canonical())
    with pytest.raises(NativeSpatialLayoutError) as caught:
        native_spatial_layouts(plan)
    refusal = caught.value
    assert refusal.code == "native_geometry_evolution_unavailable"
    assert refusal.evidence["blockage"]["classification"] == "IMPL"
    assert refusal.evidence["blockage"]["phase"] == "resolve"
    assert "state_geometry_exchange_atomic_publication_and_rollback" in \
        refusal.evidence["geometry_evolution"]["obligations"]
    with pytest.raises(LayoutCapabilityError) as public:
        resolve_native_spatial_layouts(plan)
    assert public.value.evidence["gate"] == refusal.code
    assert public.value.evidence["refusal"]["evidence"] == refusal.evidence


def test_static_provider_cannot_erase_motion_by_preserving_reference_native_shape():
    case, layout = request()
    moving = normalize_layout_plan(layout, owner=case.owner_path.canonical())
    static = normalize_layout_plan(layout.layout, owner=case.owner_path.canonical())
    assert moving.layouts[0].native_spatial_layout == static.layouts[0].native_spatial_layout
    assert moving.qualified_id != static.qualified_id
    assert native_spatial_layouts(static)
    with pytest.raises(NativeSpatialLayoutError, match="discrete GCL"):
        native_spatial_layouts(moving)


def test_map_requires_exact_frame_rank_and_no_python_callback():
    _, layout = request()
    with pytest.raises(TypeError, match="ScalarExpr"):
        GeometryEvolution((lambda x, t: x,))
    with pytest.raises(ValueError, match="frame and rank"):
        MovingControlVolumes(layout.layout, evolution=GeometryEvolution(
            layout.evolution.coordinate_map * 2))
