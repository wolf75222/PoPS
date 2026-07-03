"""Spec 5 sec.5.14 / sec.8.6 (ADC-555): case.amr.refine chains onto the AMR layout descriptor.

``case.amr`` is a thin authoring shim, NOT a separate AMR engine: ``.refine(criterion, regrid=,
nesting=, patches=)`` writes each policy straight onto the case's ``AMR`` layout and returns the
case so the call chains. This confirms every documented slot (refine / regrid / nesting /
patches) actually lands on the layout, and that the tags surface through ``layout.inspect()``
(``compiled.inspect_amr()`` carrying them is covered by
``tests/python/integration/amr/test_amr_runtime_inspect.py``).

Pure Python; needs only ``import pops`` (nothing computes on a grid).
"""
import pytest

pops = pytest.importorskip("pops")

from pops.mesh.amr import (  # noqa: E402
    PatchLayout, ProperNesting, Refine, RegridEvery, TagUnion)
from pops.mesh.cartesian import CartesianMesh  # noqa: E402
from pops.mesh.layouts import AMR  # noqa: E402


class _FakeModel:
    """A minimal model advertising its declared subjects (mirrors HyperbolicModel's surface)."""

    cons_names = ["rho"]
    cons_roles = None


def _case():
    return pops.Problem(layout=AMR(base=CartesianMesh(n=32))).block("ne", physics=_FakeModel())


def test_refine_writes_the_criterion_onto_the_layout_and_chains():
    case = _case()
    criterion = Refine.on("rho").above(0.1)
    assert case.amr.refine(criterion) is case
    assert case.layout.refine is criterion


def test_refine_chains_regrid_nesting_and_patches_in_one_call():
    case = _case()
    regrid = RegridEvery(4)
    nesting = ProperNesting(buffer=2)
    patches = PatchLayout(distribute_coarse=True, coarse_max_grid=16)
    result = case.amr.refine(Refine.on("rho").above(0.1),
                             regrid=regrid, nesting=nesting, patches=patches)
    assert result is case
    assert case.layout.regrid is regrid
    assert case.layout.nesting is nesting
    assert case.layout.patches is patches


def test_refine_slots_chain_across_separate_calls():
    # Each slot can also be set on its own call; later calls only touch the slots they pass.
    case = _case()
    case.amr.refine(regrid=RegridEvery(2))
    case.amr.refine(nesting=ProperNesting(buffer=1))
    case.amr.refine(patches=PatchLayout(coarse_max_grid=8))
    assert case.layout.regrid.steps == 2
    assert case.layout.nesting.buffer == 1
    assert case.layout.patches.coarse_max_grid == 8
    # No criterion was ever passed: refine stays unset.
    assert case.layout.refine is None


def test_refine_tag_union_surfaces_in_layout_inspect():
    case = _case()
    case.amr.refine(TagUnion(Refine.on("rho").above(0.1), Refine.on("rho").below(-0.1)),
                    regrid=RegridEvery(3))
    manifest = case.layout.inspect()
    amr_report = manifest["amr_report"]
    slots = {row["slot"] for row in amr_report["policies"]}
    assert {"refine", "regrid"} <= slots
    # The sub-criteria of the TagUnion are individually named, not just the union count.
    criterion_slots = [row for row in amr_report["policies"] if row["slot"] == "refine.criterion"]
    assert len(criterion_slots) == 2


def test_refine_rejects_a_bogus_role_before_writing_to_the_layout():
    case = _case()
    with pytest.raises(ValueError, match="is not a declared subject"):
        case.amr.refine(Refine.on("definitely_not_a_role").above(0.05))
    # The rejected criterion never lands on the layout (fail before mutate).
    assert case.layout.refine is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
