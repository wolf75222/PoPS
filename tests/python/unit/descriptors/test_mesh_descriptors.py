"""Spec 5 (sec.5.9-5.11 / sec.8): the pops.mesh typed descriptor surface.

These exercise the inert mesh / layout / AMR descriptors: their options/capabilities,
the explainable AMR route limits (max_levels / ratio), the typed refinement criteria,
the canonical ``pops.mesh`` package, and short printable summaries. Pure Python; nothing here
computes on a grid.
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.mesh import CartesianMesh, PolarMesh, AuxHalo, PatchBox, BoxLayout  # noqa: E402
from pops.mesh.layouts import Uniform, AMR  # noqa: E402
from pops.mesh.amr import (  # noqa: E402
    Refine, TagUnion, RegridEvery, FrozenRegrid, PatchLayout, ProperNesting)
from pops.mesh.geometry import Disc, EmbeddedBoundary  # noqa: E402
from pops.mesh.masks import CutCell, NoMask, Staircase  # noqa: E402
from pops.mesh.boundaries import Periodic, Physical, FaceBC, XMin  # noqa: E402
from pops.model import Handle, OwnerPath  # noqa: E402
from pops.ir.ops import dx, dy, sqrt  # noqa: E402
from pops.ir import ValueExpr, Var  # noqa: E402


def _handle(name, kind="state"):
    return Handle(name, kind=kind, owner=OwnerPath.model("mesh-descriptor-tests"))


def test_mesh_package_is_the_only_public_descriptor_path():
    assert pops.mesh.CartesianMesh is CartesianMesh
    assert pops.mesh.PolarMesh is PolarMesh
    assert "CartesianMesh" not in pops.__all__
    assert "PolarMesh" not in pops.__all__
    assert not hasattr(pops, "CartesianMesh")
    assert not hasattr(pops, "PolarMesh")


def test_cartesian_options_and_caps():
    m = CartesianMesh(n=128, L=2.0, periodic=False)
    assert m.options() == {"n": 128, "L": 2.0, "periodic": False}
    assert m.capabilities().to_dict()["geometry"] == "cartesian"
    assert m.capabilities().to_dict()["dim"] == 2


def test_dimension_policy_rejects_non_2d_meshes():
    with pytest.raises(ValueError, match="dimension=3"):
        CartesianMesh(dim=3)
    with pytest.raises(ValueError, match="dimension=3"):
        PolarMesh(0.1, 1.0, 8, 16, dim=3)


def test_polar_validation():
    PolarMesh(0.1, 1.0, 8, 16, theta_boxes=4)  # valid
    with pytest.raises(ValueError):
        PolarMesh(1.0, 0.5, 8, 16)  # r_max <= r_min
    with pytest.raises(ValueError):
        PolarMesh(0.1, 1.0, 2, 16)  # nr < 3
    with pytest.raises(ValueError):
        PolarMesh(0.1, 1.0, 8, 16, theta_boxes=5)  # 5 does not divide 16


def test_patch_box_and_layout():
    b = PatchBox(lo=(0, 0), hi=(3, 7))
    assert b.shape == (4, 8)
    with pytest.raises(ValueError):
        PatchBox(lo=(0, 0), hi=(-1, 2))
    layout = BoxLayout([b, PatchBox((4, 0), (7, 7))])
    assert len(layout) == 2


def test_uniform_layout():
    u = Uniform(CartesianMesh(), embedded_boundary=EmbeddedBoundary(Disc(), CutCell()))
    assert u.capabilities().supports("amr") is False
    assert "embedded_boundary" in u.options()


def test_amr_route_limits_are_explainable():
    m = CartesianMesh(n=128)
    ok = AMR(base=m, max_levels=3, ratio=2, regrid=RegridEvery(20),
             patches=PatchLayout(distribute_coarse=True, coarse_max_grid=32),
             nesting=ProperNesting(buffer=1))
    assert ok.available().ok
    ok.validate()
    deep = AMR(base=m, max_levels=4)
    assert deep.available().ok
    deep.validate()
    with pytest.raises(ValueError, match="ratio 3"):
        AMR(base=m, ratio=3).validate()


def test_typed_refinement_criteria():
    rho = _handle("rho")
    phi = _handle("phi", kind="field")
    c = Refine.on(rho).above(0.05)
    assert c.options()["predicate"] == "above" and c.threshold == 0.05
    assert c.options()["subject"]["handle"]["local_id"] == "rho"
    c.validate()
    with pytest.raises(ValueError):
        Refine.on(rho).validate()  # incomplete: no predicate/threshold
    TagUnion(Refine.on(rho).above(0.05),
             Refine.on(phi).gradient_above(0.5)).validate()
    with pytest.raises(TypeError, match="Handle"):
        Refine.on("rho")
    with pytest.raises(TypeError):
        TagUnion("not-a-criterion")


def test_refine_accepts_a_reference_aware_symbolic_indicator_without_flattening_it():
    rho = _handle("rho")
    indicator = sqrt(dx(ValueExpr(rho)) ** 2 + dy(ValueExpr(rho)) ** 2)
    criterion = Refine.on(indicator).above(0.05)
    resolved = criterion.resolve_references(lambda handle: handle)
    assert resolved.subject is not indicator
    assert resolved.subject.a.a.a.field.handle is rho
    options = resolved.options()["subject"]
    assert options["reference_type"] == "expression"
    assert options["expression_type"].endswith(".Sqrt")


def test_refine_expression_rejects_free_name_var_provenance():
    with pytest.raises(TypeError, match=r"ValueExpr\(handle\)"):
        Refine.on(Var("rho", "cons"))

    rho = _handle("rho")
    mixed = ValueExpr(rho) + Var("legacy_rho", "cons")
    with pytest.raises(TypeError, match="free-name Var"):
        Refine.on(mixed).above(0.1).resolve_references(lambda handle: handle)


def test_boundaries_and_masks():
    assert Periodic().capabilities().to_dict()["periodic"] is True
    assert Physical("wall").options()["kind"] == "wall"
    with pytest.raises(ValueError):
        Physical("nope")
    FaceBC(XMin(), Periodic())
    with pytest.raises(TypeError):
        FaceBC("x", Periodic())
    assert CutCell().capabilities().to_dict()["conservative"] is True
    assert NoMask().capabilities().to_dict()["masked_transport"] is False
    assert Staircase().capabilities().to_dict()["conservative"] is False


def test_amr_policies():
    assert FrozenRegrid().options()["frozen"] is True
    assert RegridEvery(20).options()["steps"] == 20
    with pytest.raises(ValueError):
        RegridEvery(0)


def test_printable_summaries_are_short_and_stable():
    s = str(AMR(base=CartesianMesh()))
    assert s.startswith("AMR") and len(s) < 200
    assert "CartesianMesh" in repr(CartesianMesh())
    assert str(Refine.on(_handle("rho")).above(0.05)).startswith("Refine")
    assert str(AuxHalo("foextrap")) == "AuxHalo('foextrap', value=0)"


# The CI python runner invokes each test file as `python3 <file>`; run pytest on this
# module so the assertions execute (a bare import would only define the test functions).
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
