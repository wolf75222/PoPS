"""Inert adaptive-layout inspection through the sole public ``pops.inspect`` dispatcher.

The report names the level / ratio envelope, attached regrid / refine / nesting policies, runtime
requirements (reflux, tag reduction), and explainable route limitations for adaptive and uniform
layouts. Checkpoint and output consumers are intentionally absent from the layout report. Pure
Python: descriptor inspection imports no ``_pops`` / runtime / codegen.
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.mesh import CartesianMesh  # noqa: E402
from pops.mesh.layouts import AMR, Uniform  # noqa: E402
from pops.mesh.amr import (  # noqa: E402
    Refine, TagUnion, RegridEvery, ProperNesting, PatchLayout, NATIVE_RATIOS)
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
        nesting=ProperNesting(buffer=1))


def test_layout_inspection_uses_the_generic_public_dispatcher():
    assert "inspect" in pops.__all__
    assert not hasattr(pops, "inspect_amr")
    assert pops.inspect(_full_amr()) == _full_amr().inspect()


def test_amr_layout_report_levels_ratio_and_policies():
    d = pops.inspect(_full_amr())["amr_report"]
    assert d["layout"] == "amr"
    assert d["max_levels"] == 3 and d["ratio"] == 2
    assert d["available"] == "yes"
    # The reflux / tag-reduction runtime requirements come straight from the descriptor.
    assert d["requirements"]["reflux"] is True
    assert d["requirements"]["tag_reduction"] is True
    assert d["requirements"]["amr_runtime"] is True
    # Every attached policy slot is reported, and the TagUnion expands into its criteria.
    slots = [p["slot"] for p in d["policies"]]
    for slot in ("refine", "regrid", "patches", "nesting"):
        assert slot in slots, "policy slot %r missing from the AMR report" % slot
    assert slots.count("refine.criterion") == 2
    names = {p["slot"]: p["name"] for p in d["policies"]}
    assert names["regrid"] == "RegridEvery"
    assert names["refine"] == "TagUnion"
    # The expanded criterion rows name each tagged subject (not just the union count).
    crit_subjects = " ".join(
        str(p["options"]) for p in d["policies"] if p["slot"] == "refine.criterion")
    assert "rho" in crit_subjects and "phi" in crit_subjects


def test_amr_report_is_deterministic():
    first = pops.inspect(_full_amr())
    second = pops.inspect(_full_amr())
    assert first == second
    assert first["amr_report"]["max_levels"] == 3
    assert first["amr_report"]["native_max_levels"] == "resource_policy"


def test_arbitrary_depth_is_reported_as_resource_policy():
    report = pops.inspect(AMR(base=CartesianMesh(n=128), max_levels=4))["amr_report"]
    assert report["available"] == "yes"
    assert report["max_levels"] == 4
    assert report["native_max_levels"] == "resource_policy"
    assert any("resource-policy" in note for note in report["limitations"])


def test_uniform_layout_reports_single_level():
    d = pops.inspect(Uniform(CartesianMesh()))["amr_report"]
    assert d["layout"] == "uniform"
    assert d["max_levels"] == 1
    assert d["policies"] == []
    assert any("single-level" in note for note in d["limitations"])
# The CI python runner invokes each test file as `python3 <file>`; run pytest on this
# module so the assertions execute (a bare import would only define the test functions).
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
