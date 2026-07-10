"""Spec 5 sec.8.6 / sec.5.14 (ADC-589, ADC-555): Uniform + active AMR criteria is refused.

A refinement criterion attached to a single-level ``Uniform`` layout has no level to refine
onto. Silently dropping it would be a correctness trap (the user believes the tag is active), so
:meth:`pops.problem.Problem.validate` refuses it by default and only accepts the explicit
``Uniform(mesh, refine=..., ignore_amr=pops.mesh.amr.IgnoreAMRCriteria())`` escape.

Pure Python; needs only ``import pops`` (nothing computes on a grid).
"""
import pytest

pops = pytest.importorskip("pops")

from pops.mesh.amr import IgnoreAMRCriteria, Refine, TagUnion  # noqa: E402
from pops.mesh.cartesian import CartesianMesh  # noqa: E402
from pops.mesh.layouts import AMR, Uniform  # noqa: E402
from pops.model import DeclarationIndex, Handle, OwnerKind, OwnerPath  # noqa: E402


_RHO = Handle("rho", kind="state", owner=OwnerPath.shared("mesh.uniform_refine"))


def _refine():
    return Refine.on(_RHO)


class _FakeModel:
    """A minimal model advertising its declared subjects (mirrors HyperbolicModel's surface)."""

    def __init__(self):
        self.name = "uniform-model"
        self.owner_path = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, self.name)

    def declaration_index(self):
        return DeclarationIndex(owner=self.owner_path, handles=())


def _case(layout):
    return pops.Problem(layout=layout).block("ne", physics=_FakeModel())


def test_uniform_without_refine_validates():
    # Baseline: a plain Uniform layout (no criterion attached) is untouched by the new refusal;
    # Problem.validate() raises loud on error and returns normally otherwise.
    case = _case(Uniform(CartesianMesh(n=16)))
    case.validate()


def test_uniform_plus_refine_is_refused_by_default():
    layout = Uniform(CartesianMesh(n=16), refine=_refine().above(0.1))
    case = _case(layout)
    with pytest.raises(ValueError, match="carries active AMR criteria"):
        case.validate()


def test_uniform_plus_tag_union_is_refused_by_default():
    layout = Uniform(
        CartesianMesh(n=16),
        refine=TagUnion(_refine().above(0.1), _refine().below(-0.1)))
    case = _case(layout)
    with pytest.raises(ValueError) as exc:
        case.validate()
    msg = str(exc.value)
    assert "IgnoreAMRCriteria" in msg
    assert "layout=AMR(...)" in msg


def test_uniform_plus_refine_with_ignore_amr_criteria_passes():
    layout = Uniform(
        CartesianMesh(n=16),
        refine=_refine().above(0.1),
        ignore_amr=IgnoreAMRCriteria())
    case = _case(layout)
    # No raise: the explicit escape is honoured.
    case.validate()
    # The escape is visible on the layout's own inspection, not swallowed.
    opts = layout.options()
    assert opts["refine"] == "Refine"
    assert opts["ignore_amr"] is True


def test_amr_layout_is_unaffected_by_the_uniform_refusal():
    # An AMR layout with a refine criterion is the actual target route: never refused here.
    layout = AMR(base=CartesianMesh(n=16), refine=_refine().above(0.1))
    case = _case(layout)
    case.validate()


def test_ignore_amr_requires_the_typed_marker():
    # The escape is the typed descriptor, never a free truthy value: ignore_amr=True would be
    # an untyped opt-out and is refused at construction.
    with pytest.raises(TypeError, match="IgnoreAMRCriteria"):
        Uniform(CartesianMesh(n=16), refine=_refine().above(0.1), ignore_amr=True)


def test_refine_subject_rejects_a_string():
    with pytest.raises(TypeError, match="names and strings"):
        Refine.on("rho")


def test_ignore_amr_criteria_is_a_plain_marker_descriptor():
    marker = IgnoreAMRCriteria()
    assert marker.category == "amr_override"
    assert marker.options() == {"ignore_amr_criteria": True}
    assert marker.validate() is True


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
