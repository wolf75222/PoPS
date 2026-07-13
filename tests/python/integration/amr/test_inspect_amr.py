"""Spec 5 (sec.5.11 / sec.8): the inert pops.inspect_amr authoring report.

These exercise :func:`pops.inspect_amr`, the descriptor-sourced report of an AMR hierarchy
(the introspectable counterpart of :func:`pops.inspect_capabilities` for the adaptive-mesh
route). They assert the report names the levels / ratio envelope, the attached regrid /
refine / nesting / checkpoint / output policies, the runtime requirements (reflux, tag
reduction) and the explainable route limitations, for the native envelope (``None``), a full
AMR layout, a deeper resource-policy-controlled layout, and a Uniform layout. Pure Python: only
``import pops`` is needed (nothing computes on a grid); inspect_amr reads descriptor metadata
and imports no ``_pops`` / runtime / codegen.
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.mesh import CartesianMesh  # noqa: E402
from pops.mesh.layouts import AMR, Uniform  # noqa: E402
from pops.mesh.amr import (  # noqa: E402
    Refine, TagUnion, RegridEvery, ProperNesting, PatchLayout, CheckpointPolicy,
    AMROutput, AllLevels, NATIVE_RATIOS)
from pops.model import Handle, OwnerPath  # noqa: E402


def _ref(name, kind="state"):
    return Handle(name, kind=kind, owner=OwnerPath.shared("inspect-amr"))


def _full_amr():
    """A fully-policied three-level AMR layout."""
    return AMR(
        base=CartesianMesh(n=128), max_levels=3, ratio=NATIVE_RATIOS[0],
        regrid=RegridEvery(20),
        patches=PatchLayout(distribute_coarse=True, coarse_max_grid=32),
        refine=TagUnion(Refine.on(_ref("rho")).above(0.05),
                        Refine.on(_ref("phi", kind="field")).gradient_above(0.5)),
        nesting=ProperNesting(buffer=1),
        checkpoint=CheckpointPolicy(restartable=True),
        output=AMROutput(fields=[_ref("phi", kind="field")], levels=AllLevels(),
                         include_patch_boxes=True))


def test_inspect_amr_is_exported_at_top_level():
    assert hasattr(pops, "inspect_amr")
    assert "inspect_amr" in pops.__all__


def test_native_envelope_report_when_none():
    rep = pops.inspect_amr()
    d = rep.to_dict()
    assert d["layout"] == "native-envelope"
    assert d["max_levels"] == "resource_policy"
    assert d["native_max_levels"] == "resource_policy"
    assert d["ratio"] == NATIVE_RATIOS[0]
    assert d["available"] == "yes"
    assert any("resource-policy" in note for note in d["limitations"])


def test_amr_layout_report_levels_ratio_and_policies():
    rep = pops.inspect_amr(_full_amr())
    d = rep.to_dict()
    assert d["layout"] == "amr"
    assert d["max_levels"] == 3 and d["ratio"] == 2
    assert d["available"] == "yes"
    # The reflux / tag-reduction runtime requirements come straight from the descriptor.
    assert d["requirements"]["reflux"] is True
    assert d["requirements"]["tag_reduction"] is True
    assert d["requirements"]["amr_runtime"] is True
    # Every attached policy slot is reported, and the TagUnion expands into its criteria.
    slots = [p["slot"] for p in d["policies"]]
    for slot in ("refine", "regrid", "patches", "nesting", "checkpoint", "output"):
        assert slot in slots, "policy slot %r missing from the AMR report" % slot
    assert slots.count("refine.criterion") == 2
    names = {p["slot"]: p["name"] for p in d["policies"]}
    assert names["regrid"] == "RegridEvery"
    assert names["refine"] == "TagUnion"
    # The expanded criterion rows name each tagged subject (not just the union count).
    crit_subjects = " ".join(
        str(p["options"]) for p in d["policies"] if p["slot"] == "refine.criterion")
    assert "rho" in crit_subjects and "phi" in crit_subjects


def test_amr_report_print_is_short_and_deterministic():
    rep = pops.inspect_amr(_full_amr())
    text = str(rep)
    assert text.startswith("AMR hierarchy report")
    assert "max_levels=3" in text and "ratio=2" in text
    assert "native depth: resource_policy" in text
    assert "reflux" in text
    assert "RegridEvery(steps=20)" in text
    # The report is deterministic (same input -> same string).
    assert str(pops.inspect_amr(_full_amr())) == text
    assert len(text) < 1200


def test_arbitrary_depth_is_reported_as_resource_policy():
    rep = pops.inspect_amr(AMR(base=CartesianMesh(n=128), max_levels=4))
    assert rep.available == "yes"
    assert rep.max_levels == 4
    assert rep.native_max_levels == "resource_policy"
    assert any("resource-policy" in note for note in rep.limitations)


def test_uniform_layout_reports_single_level():
    rep = pops.inspect_amr(Uniform(CartesianMesh()))
    d = rep.to_dict()
    assert d["layout"] == "uniform"
    assert d["max_levels"] == 1
    assert d["policies"] == []
    assert any("single-level" in note for note in d["limitations"])


def test_inspect_amr_rejects_a_non_layout():
    with pytest.raises(TypeError):
        pops.inspect_amr("amr")


# The CI python runner invokes each test file as `python3 <file>`; run pytest on this
# module so the assertions execute (a bare import would only define the test functions).
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
