from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import get_type_hints

import numpy as np
import pytest

from pops.domain import DomainPreview, Rectangle
from pops.domain.preview import GeometryPreviewProvider
from pops.mesh.geometry import Disc


def test_rectangle_preview_is_typed_read_only_data_without_geometry() -> None:
    domain = Rectangle("box", (-2.0, -1.0), (2.0, 3.0))

    preview = domain.preview(resolution=(9, 5))

    assert isinstance(preview, DomainPreview)
    assert preview.domain is domain
    assert preview.geometry is None
    assert preview.resolution == (9, 5)
    assert preview.x[[0, -1]].tolist() == [-2.0, 2.0]
    assert preview.y[[0, -1]].tolist() == [-1.0, 3.0]
    assert preview.level_set_values is None
    assert preview.active_mask is None
    assert not preview.x.flags.writeable
    assert not preview.y.flags.writeable


def test_csg_preview_uses_the_generic_level_set_sign_convention() -> None:
    domain = Rectangle("unit_square", (0.0, 0.0), (1.0, 1.0))
    annulus = Disc(center=(0.5, 0.5), radius=0.35) \
        - Disc(center=(0.5, 0.5), radius=0.15)

    preview = domain.preview(geometry=annulus, resolution=101)

    assert preview.resolution == (101, 101)
    assert preview.level_set_values is not None
    assert preview.active_mask is not None
    assert preview.active_mask.shape == (101, 101)
    assert not preview.active_mask[50, 50]
    assert preview.active_mask[50, 75]
    assert not preview.active_mask[0, 0]
    assert np.array_equal(preview.active_mask, preview.level_set_values < 0.0)
    assert not preview.level_set_values.flags.writeable
    assert not preview.active_mask.flags.writeable


def test_preview_accepts_a_structural_third_party_geometry_provider() -> None:
    class ThirdPartyDisc:
        def level_set(self, frame: object) -> object:
            return Disc(center=(0.5, 0.5), radius=0.2).level_set(frame)

    domain = Rectangle("unit_square", (0.0, 0.0), (1.0, 1.0))

    preview = domain.preview(geometry=ThirdPartyDisc(), resolution=21)

    assert preview.active_mask is not None
    assert preview.active_mask[10, 10]
    assert not preview.active_mask[0, 0]


def test_rectangle_preview_annotations_resolve_for_autodoc() -> None:
    preview_hints = get_type_hints(Rectangle.preview)
    show_hints = get_type_hints(Rectangle.show)

    assert preview_hints["return"] is DomainPreview
    assert preview_hints["geometry"] == GeometryPreviewProvider | None
    assert show_hints["return"] == Path | None


def test_public_preview_value_rejects_non_finite_axes_and_inconsistent_mask() -> None:
    domain = Rectangle("unit_square", (0.0, 0.0), (1.0, 1.0))

    with pytest.raises(ValueError, match="finite"):
        DomainPreview(domain, None, np.array([0.0, np.nan]), np.array([0.0, 1.0]))

    geometry = Disc(center=(0.5, 0.5), radius=0.25)
    values = np.array([[-1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="active_mask"):
        DomainPreview(
            domain,
            geometry,
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            values,
            np.ones((2, 2), dtype=np.bool_),
        )


@pytest.mark.parametrize("resolution", (True, 1, (8,), (8, 1), (8, 4.5)))
def test_preview_rejects_ambiguous_or_too_small_resolutions(resolution: object) -> None:
    domain = Rectangle("box", (0.0, 0.0), (1.0, 1.0))

    with pytest.raises((TypeError, ValueError), match="resolution"):
        domain.preview(resolution=resolution)  # type: ignore[arg-type]


def test_preview_sampling_does_not_import_matplotlib() -> None:
    root = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "python")
    program = """
import sys
from pops.domain import Rectangle
from pops.mesh.geometry import Disc
domain = Rectangle('box', (0, 0), (1, 1))
domain.preview(geometry=Disc(radius=0.25), resolution=8)
assert 'matplotlib' not in sys.modules
assert 'matplotlib.pyplot' not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_show_with_path_saves_without_opening_an_interactive_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    def forbidden_show() -> None:
        raise AssertionError("Rectangle.show(path=...) must not open a window")

    monkeypatch.setattr(plt, "show", forbidden_show)
    target = tmp_path / "nested" / "domain.svg"
    domain = Rectangle("box", (0.0, 0.0), (2.0, 1.0))

    result = domain.show(
        geometry=Disc(center=(1.0, 0.5), radius=0.3),
        resolution=(40, 20),
        path=target,
    )

    assert result == target
    assert target.is_file()
    assert "<svg" in target.read_text(encoding="utf-8")[:500]
