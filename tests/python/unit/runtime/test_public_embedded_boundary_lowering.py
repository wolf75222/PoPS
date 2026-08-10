from __future__ import annotations

import inspect
from typing import Any

import pytest

from pops.analytic import coordinates
from pops.boundary import ZeroFlux
from pops.domain import CartesianDomain
from pops.layouts import AMR, Uniform
from pops.mesh import CartesianGrid, normalize_layout_plan
from pops.mesh.geometry import EmbeddedBoundary, Geometry, LevelSet
from pops.mesh.masks import CutCell, Staircase
from pops.model import OwnerPath
from pops.runtime._runtime_mesh_lowering import install_embedded_boundary
from pops.runtime import _runtime_executor
from tests.python.support.layout_plan import final_amr_layout


def _ranked_grid(dimension: int) -> CartesianGrid:
    frame = CartesianDomain(
        "embedded-rank-%d" % dimension,
        (0.0,) * dimension,
        (1.0,) * dimension,
    ).frame()
    return CartesianGrid(frame=frame, cells=(8,) * dimension)


def _ranked_embedded_boundary(grid: CartesianGrid) -> EmbeddedBoundary:
    values = coordinates(grid.frame)
    expression = values[0]
    for value in values[1:]:
        expression = expression + value
    return EmbeddedBoundary(
        LevelSet(expression - 0.5 * len(values)),
        CutCell(kappa_min=0.05, face_open_eps=0.01, cut_theta_min=0.02),
        ZeroFlux(),
    )


def _adaptive_layout(
    grid: CartesianGrid, embedded_boundary: EmbeddedBoundary,
) -> AMR:
    base = final_amr_layout(grid, max_levels=2, ratio=2)
    return AMR(
        grid=grid,
        hierarchy=base.hierarchy,
        tagging=base.tagging,
        regrid=base.regrid,
        transfer=base.transfer,
        execution=base.execution,
        embedded_boundary=embedded_boundary,
        patch_layout=base.patch_layout,
        load_balance=base.load_balance,
        tagger=base.tagger,
        clustering=base.clustering,
        reflux=base.reflux,
    )


class _NativeLevelSetProbe:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def _set_analytic_level_set(self, *values: Any) -> None:
        self.calls.append(values)


class _RuntimeProbe:
    def __init__(self) -> None:
        self._s = _NativeLevelSetProbe()


@pytest.mark.parametrize(
    "provider",
    (_runtime_executor._UniformNativeProvider, _runtime_executor._AdaptiveNativeProvider),
)
def test_native_executors_install_signed_geometry_before_compiled_blocks(provider: type) -> None:
    source = inspect.getsource(provider.install)
    assert source.index("install_embedded_boundary(engine") < source.index(
        "engine._install_compiled("
    )


@pytest.mark.parametrize("dimension", (1, 2, 3))
@pytest.mark.parametrize("adaptive", (False, True), ids=("uniform", "amr"))
def test_public_layouts_share_exact_rank_embedded_boundary_lowering(
    dimension: int, adaptive: bool,
) -> None:
    grid = _ranked_grid(dimension)
    embedded = _ranked_embedded_boundary(grid)
    layout = _adaptive_layout(grid, embedded) if adaptive else Uniform(
        grid, embedded_boundary=embedded
    )
    plan = normalize_layout_plan(
        layout,
        owner=OwnerPath.case("embedded-%s-%dd" % ("amr" if adaptive else "uniform", dimension)),
    )
    normalized, = plan.layouts

    signed = normalized.to_data()["options"]["embedded_boundary"]
    assert signed["level_set"]["active_when"] == "phi<0"
    assert signed["transport"] == {
        "mode": "cutcell",
        "kappa_min": 0.05,
        "face_open_eps": 0.01,
        "cut_theta_min": 0.02,
    }
    assert normalized.capabilities["embedded_boundary"] is True
    assert normalized.capabilities["embedded_boundary_transport"] == "cutcell"

    runtime = _RuntimeProbe()
    install_embedded_boundary(runtime, normalized)

    assert len(runtime._s.calls) == 1
    opcodes, literals, mode, kappa_min, face_open_eps, cut_theta_min = runtime._s.calls[0]
    assert isinstance(opcodes, list) and opcodes
    assert isinstance(literals, list)
    assert (mode, kappa_min, face_open_eps, cut_theta_min) == (
        "cutcell", 0.05, 0.01, 0.02,
    )


def test_resolved_amr_never_recalls_mutable_embedded_geometry() -> None:
    class MutableGeometry(Geometry):
        def __init__(self) -> None:
            self.offset = 0.25
            self.calls = 0

        def level_set(self, frame: Any) -> LevelSet:
            self.calls += 1
            return LevelSet(coordinates(frame)[0] - self.offset)

    grid = _ranked_grid(3)
    geometry = MutableGeometry()
    layout = _adaptive_layout(
        grid,
        EmbeddedBoundary(geometry, Staircase(), ZeroFlux()),
    )
    authored = layout.options()["embedded_boundary"]
    assert geometry.calls == 2

    geometry.offset = 0.75
    resolved = layout.resolve_for_case(lambda value: value)
    resolved.validate()
    plan = normalize_layout_plan(
        resolved, owner=OwnerPath.case("resolved-mutable-amr-embedded")
    )

    assert geometry.calls == 2
    assert resolved.embedded_boundary is None
    assert resolved.options()["embedded_boundary"] == authored
    assert resolved.capabilities().get("embedded_boundary_transport") == "staircase"
    assert plan.layouts[0].to_data()["options"]["embedded_boundary"] == authored
