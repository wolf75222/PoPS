"""ADC-527: pops.inspect(obj) is a stable, serialisable dispatcher over any descriptor / Problem /
report. It dispatches to obj.inspect() and is distinct from pops.inspect_capabilities/inspect_amr.

Pure Python; needs only `import pops`.
"""
import json
import sys

import pytest

pops = pytest.importorskip("pops")


class _StubModel:
    name = "m"


def test_pops_inspect_is_public():
    assert hasattr(pops, "inspect")
    assert "inspect" in pops.__all__
    # Distinct from the native capability matrix entry points.
    assert pops.inspect is not pops.inspect_capabilities
    assert pops.inspect is not pops.inspect_amr


def test_inspect_dispatches_to_a_descriptor():
    from pops.mesh.cartesian import CartesianMesh
    mesh = CartesianMesh(n=8)
    record = pops.inspect(mesh)
    assert record == mesh.inspect()      # dispatches to obj.inspect()
    assert record["name"] == "CartesianMesh"


def test_inspect_a_problem_is_json_ready():
    prob = pops.Problem(name="plasma").block("ne", physics=_StubModel())
    record = pops.inspect(prob)
    assert record["name"] == "plasma"
    assert "blocks" in record
    json.dumps(record)                   # serialisable


def test_problem_inspect_is_a_typed_report_bridged_by_pops_inspect():
    # ADC-564: Problem.inspect() is a typed pops.Report (attributes), and pops.inspect(obj) is the
    # explicit dict bridge over its to_dict() -- so a structure-wanting caller reads attributes.
    from pops._report import Report
    prob = pops.Problem(name="plasma").block("ne", physics=_StubModel())
    report = prob.inspect()
    assert isinstance(report, Report) and not isinstance(report, dict)
    assert report.name == "plasma"                 # attribute access
    assert pops.inspect(prob) == report.to_dict()  # pops.inspect(obj) == report.to_dict()


def test_inspect_a_brick_descriptor():
    from pops.numerics.riemann import HLL
    brick = HLL()
    record = pops.inspect(brick)
    assert record["name"] == brick.name
    assert record["category"] == "riemann"
    assert "requirements" in record and "capabilities" in record


def test_inspect_a_validation_report():
    from pops.descriptors import ValidationReport
    report = ValidationReport().error("block", "no_model", "block ne has no model")
    record = pops.inspect(report)
    assert record["ok"] is False
    assert record["issues"][0]["family"] == "block"


def test_inspect_never_runs_numerics_on_a_plain_object():
    # A non-descriptor object falls back to a repr view, never touches the runtime.
    record = pops.inspect(object())
    assert "repr" in record


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
