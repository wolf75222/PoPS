"""The historical 2-D polar ``System`` bypass is retired fail-closed."""

from pathlib import Path

from pops.mesh import PolarMesh


def test_direct_polar_system_bypass_refuses_before_native_allocation():
    source = (
        Path(__file__).resolve().parents[4] / "python/pops/runtime/_system.py"
    ).read_text(encoding="utf-8")

    rejection = source.split("if mesh is not None:", 1)[1].split(
        "_threading._first_system_built", 1
    )[0]
    assert "raise NotImplementedError" in rejection
    assert "System(mesh=...) was retired" in rejection
    assert "System<Dim>" in rejection


def test_polar_mesh_remains_an_inert_geometry_output_descriptor():
    mesh = PolarMesh(r_min=0.3, r_max=1.0, nr=8, ntheta=16, theta_boxes=4)
    geometry = mesh.normalized_geometry()

    assert geometry.coordinate_system == "pops://coordinates/polar-annulus-2d@1"
    assert geometry.cell_measure == "pops://cell-measures/polar-annulus-area@1"
    assert mesh.capabilities().to_dict()["native_execution"] is False
    assert not hasattr(mesh, "_apply_system_config")
