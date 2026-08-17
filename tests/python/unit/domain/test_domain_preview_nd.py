from __future__ import annotations

from pathlib import Path
from typing import get_type_hints

import pytest

from pops.analytic import constant, x, y, z
from pops.domain import CartesianDomain, DomainPreview, preview_domain, preview_geometry
from pops.frames import Cartesian, Cartesian1D, Cartesian2D, Cartesian3D
from pops.mesh.geometry import LevelSet


def test_cartesian_domain_preview_samples_every_compiled_rank():
    line = CartesianDomain("line", (0.0,), (2.0,))
    plane = CartesianDomain("plane", (0.0, -1.0), (1.0, 1.0))
    box = CartesianDomain("box", (0.0, 0.0, 0.0), (1.0, 2.0, 3.0))

    line_preview = line.preview(resolution=9)
    plane_preview = plane.preview(resolution=(8, 5))
    box_preview = box.preview(resolution=(6, 5, 4))

    assert line_preview.dimension == 1
    assert line_preview.resolution == (9,)
    assert line_preview.y is None and line_preview.z is None
    assert line_preview.x[[0, -1]].tolist() == [0.0, 2.0]
    assert plane_preview.dimension == 2
    assert plane_preview.resolution == (8, 5)
    assert plane_preview.z is None
    assert box_preview.dimension == 3
    assert box_preview.resolution == (6, 5, 4)
    assert box_preview.z[[0, -1]].tolist() == [0.0, 3.0]


def test_cartesian_preview_evaluates_ranked_analytic_coordinates():
    line = CartesianDomain("line", (-1.0,), (1.0,))
    box = CartesianDomain("box", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    line_preview = line.preview(field=x(line.frame()), resolution=5)
    box_preview = box.preview(
        field=x(box.frame()) + y(box.frame()) + z(box.frame()),
        resolution=(5, 5, 5),
    )

    assert line_preview.field_values is not None
    assert line_preview.field_values.shape == (5,)
    assert line_preview.field_values[0] == pytest.approx(-1.0)
    assert line_preview.field_values[-1] == pytest.approx(1.0)
    assert box_preview.field_values is not None
    assert box_preview.field_values.shape == (5, 5, 5)
    assert box_preview.field_values[0, 0, 0] == pytest.approx(0.0)
    assert box_preview.field_values[-1, -1, -1] == pytest.approx(3.0)


def test_cartesian_preview_binds_the_domain_owned_frame_not_a_hardcoded_cartesian2d():
    domain = CartesianDomain("box", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    frame = domain.frame()
    preview = preview_domain(domain, field=z(frame), resolution=(4, 4, 4))

    assert frame.coordinates.dimension == 3
    assert not isinstance(frame.coordinates, Cartesian2D)
    assert preview.field_values is not None
    assert preview.field_values[0, 0, 0] == pytest.approx(0.0)
    assert preview.field_values[-1, 0, 0] == pytest.approx(1.0)


def test_one_dimensional_frame_has_no_transverse_coordinates():
    domain = CartesianDomain("line", (0.0,), (1.0,))
    frame = domain.frame()
    with pytest.raises((AttributeError, TypeError), match="y"):
        y(frame)


def test_three_dimensional_level_set_preview_uses_a_ranked_sample_grid():
    domain = CartesianDomain("box", (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    frame = domain.frame()
    sphere = LevelSet(x(frame) ** 2 + y(frame) ** 2 + z(frame) ** 2 - 0.25)

    preview = domain.preview(geometry=sphere, resolution=(9, 9, 9))

    assert preview.active_mask is not None
    assert preview.active_mask.shape == (9, 9, 9)
    assert preview.active_mask[4, 4, 4]
    assert not preview.active_mask[0, 0, 0]


@pytest.mark.parametrize(
    ("domain", "coordinates"),
    (
        (CartesianDomain("line", (0.0,), (1.0,)), Cartesian1D()),
        (CartesianDomain("plane", (0.0, 0.0), (1.0, 1.0)), Cartesian2D()),
        (CartesianDomain("box", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), Cartesian3D()),
    ),
)
def test_cartesian_domain_frame_accepts_the_matching_rank_constructor(domain, coordinates):
    frame = domain.frame(coordinates)
    assert frame.coordinates.dimension == domain.dimension
    assert frame.coordinates.axes == coordinates.axes


def test_cartesian_domain_frame_defaults_to_the_domain_rank():
    line = CartesianDomain("line", (0.0,), (1.0,))
    plane = CartesianDomain("plane", (0.0, 0.0), (1.0, 1.0))
    box = CartesianDomain("box", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    assert line.frame().coordinates == Cartesian(1)
    assert plane.frame().coordinates == Cartesian(2)
    assert box.frame().coordinates == Cartesian(3)
    assert line.frame().coordinates.axes == Cartesian1D().axes
    assert plane.frame().coordinates.axes == Cartesian2D().axes
    assert box.frame().coordinates.axes == Cartesian3D().axes
    with pytest.raises(TypeError, match="Cartesian"):
        box.frame("cartesian-3d")
    with pytest.raises(ValueError, match="ranks differ"):
        box.frame(Cartesian2D())


def test_preview_geometry_samples_explicit_one_and_three_dimensional_extents():
    class AlwaysActive:
        def level_set(self, frame: object) -> object:
            del frame
            return LevelSet(constant(-1.0))

    geometry = AlwaysActive()
    line = preview_geometry(geometry, extent=((0.0,), (2.0,)), resolution=8)
    box = preview_geometry(
        geometry, extent=((0.0, 0.0, 0.0), (1.0, 1.0, 2.0)), resolution=(4, 5, 6),
    )

    assert line.dimension == 1
    assert line.resolution == (8,)
    assert line.y is None and line.z is None
    assert line.active_mask is not None and bool(line.active_mask.all())
    assert box.dimension == 3
    assert box.resolution == (4, 5, 6)
    assert box.z[[0, -1]].tolist() == [0.0, 2.0]
    assert box.active_mask is not None and box.active_mask.shape == (6, 5, 4)


def test_cartesian_domain_preview_annotations_resolve_for_autodoc():
    preview_hints = get_type_hints(CartesianDomain.preview)
    show_hints = get_type_hints(CartesianDomain.show)

    assert preview_hints["return"] is DomainPreview
    assert show_hints["return"] == Path | None


def test_ranked_cartesian_show_exports_without_opening_a_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    def forbidden_show() -> None:
        raise AssertionError("CartesianDomain.show(path=...) must not open a window")

    monkeypatch.setattr(plt, "show", forbidden_show)
    line = CartesianDomain("line", (0.0,), (2.0,))
    plane = CartesianDomain("plane", (0.0, 0.0), (1.0, 1.0))
    box = CartesianDomain("box", (0.0, 0.0, 0.0), (1.0, 1.0, 2.0))
    line_target = tmp_path / "line.png"
    plane_target = tmp_path / "plane.png"
    box_target = tmp_path / "box.png"

    line_result = line.show(path=line_target, resolution=16)
    plane_result = plane.show(
        field=x(plane.frame()) + y(plane.frame()),
        resolution=(12, 8),
        path=plane_target,
    )
    box_result = box.show(
        field=x(box.frame()) + z(box.frame()),
        resolution=(8, 8, 8),
        path=box_target,
    )

    assert line_result == line_target
    assert plane_result == plane_target
    assert box_result == box_target
    for target in (line_target, plane_target, box_target):
        assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert target.stat().st_size > 0
