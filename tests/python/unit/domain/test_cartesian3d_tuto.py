"""The 3D scalar-advection tutorial uses the ranked Cartesian domain API."""

from pathlib import Path


_TUTO = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "tutorials"
    / "scalar_advection"
    / "16_openmp_cartesian3d_ssprk2.py"
)


def test_cartesian3d_tuto_uses_ranked_cartesian_domain_without_execution_context():
    source = _TUTO.read_text(encoding="utf-8")

    assert "CartesianDomain" in source
    assert "Cartesian3D" in source
    assert "from pops.domain import Rectangle" not in source
    assert "ExecutionContext" not in source
    assert "--dim 3" in source
    assert "POPS_NATIVE_DIM=3" in source
    assert "boundaries.z_min" in source
    assert "boundaries.z_max" in source
    assert "(1, NZ, NY, NX)" in source
    assert "HyQMOM" not in source
