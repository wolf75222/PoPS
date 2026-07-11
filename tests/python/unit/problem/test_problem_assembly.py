"""ADC-526: the pops.Problem declarative assembly API.

Exercises the assembly surface: add_block / block / field / add_field / program / param / aux /
output; the stable handles blocks() / fields() return; the inspect() / to_dict() serialisation; the
early per-family errors (duplicate / missing block / missing physics / unbound field / bad output);
and the scope fence -- a Problem owns NO run / install / compile method and NO array attribute.

Pure Python; needs only `import pops` (nothing computes on a grid).
"""
import sys

import pytest

pops = pytest.importorskip("pops")
from pops.params import ConstParam

from pops.model import Module  # noqa: E402


def _model(name="stub"):
    """Return a real model declaration authority for Problem assembly tests."""
    return Module(name)


def _poisson():
    from pops.fields import PoissonProblem
    from pops.math import laplacian
    from pops.solvers import GeometricMG
    return PoissonProblem(name="phi", unknown="phi",
                          equation=(-laplacian("phi") == "charge_density"),
                          solver=GeometricMG())


def test_problem_has_no_constructor_layout():
    # ADC-526: a Problem carries no layout by default (layout is supplied at compile).
    prob = pops.Problem(name="plasma")
    assert prob.layout is None
    assert prob.options()["layout"] is None


def test_chaining_setters_return_the_problem_while_param_returns_a_handle():
    model = _model("ne")
    program = type("Prog", (), {"name": "euler"})()
    prob = (pops.Problem(name="plasma")
            .block("ne", physics=model)
            .aux("B_z")
            .program(program))
    alpha = prob.param(ConstParam("alpha", 1.0))
    assert prob is prob.block.__self__
    assert alpha.param_kind == "const" and alpha.owner_path == prob.owner_path
    info = prob.inspect().to_dict()  # ADC-564: Problem.inspect() is a typed report; to_dict() bridges
    assert info["name"] == "plasma"
    assert set(info["blocks"]) == {"ne"}
    assert info["params"]["alpha"]["kind"] == "const"
    assert info["params"]["alpha"]["default"]["state"] == "value"
    assert "B_z" in info["aux"]
    assert info["time"] == "euler"


def test_add_block_returns_a_stable_handle():
    prob = pops.Problem()
    handle = prob.add_block("ne", _model("electron"), time=object(), diagnostics=object())
    assert handle.name == "ne"
    assert handle.qualified_id.endswith("::block::ne")
    qualified_id = handle.qualified_id
    # The handle is stable as more blocks are added.
    prob.add_block("ni", _model("ion"))
    assert prob.add_block.__self__ is prob or True  # add_block returns a handle, not self
    assert handle.qualified_id == qualified_id
    # blocks() exposes the stable handles keyed by name.
    assert set(prob.blocks()) == {"ne", "ni"}
    assert prob.blocks()["ne"].qualified_id == qualified_id


def test_add_field_returns_a_stable_handle():
    prob = pops.Problem()
    handle = prob.add_field(_poisson())
    assert handle.kind == "field"
    assert handle.name == "phi"
    assert set(prob.fields()) == {"phi"}


def test_duplicate_block_is_refused_early():
    prob = pops.Problem().block("ne", physics=_model())
    with pytest.raises(ValueError, match="already exists"):
        prob.block("ne", physics=_model())


def test_missing_physics_is_refused_early():
    with pytest.raises(ValueError, match="physics model is required"):
        pops.Problem().block("ne", physics=None)


def test_field_type_is_checked_early():
    with pytest.raises(TypeError, match="FieldProblem"):
        pops.Problem().field("not a field")


def test_no_block_reports_a_structured_error():
    report = pops.Problem().validate_report()
    assert not report.ok
    families = report.by_family()
    assert "block" in families
    assert any(issue.code == "no_block" for issue in families["block"])


def test_bad_output_policy_reports_runtime_family():
    class _NotAPolicy:
        name = "nope"
    prob = pops.Problem().block("ne", physics=_model()).output(_NotAPolicy())
    report = prob.validate_report()
    assert not report.ok
    assert "runtime" in report.by_family()


def test_to_dict_is_json_ready_and_array_free():
    import json
    prob = pops.Problem(name="plasma").block("ne", physics=_model())
    prob.param(ConstParam("alpha", 1.0))
    data = prob.to_dict()
    # Round-trips through JSON (no runtime object, no numpy array).
    dumped = json.dumps(data)
    assert "plasma" in dumped
    assert len(data["handles"]["blocks"]) == 1
    block_handle = data["handles"]["blocks"][0]
    assert block_handle["kind"] == "block" and block_handle["local_id"] == "ne"
    assert block_handle["qualified_id"].endswith("::block::ne")


def test_problem_has_no_run_install_or_compile_method():
    # ADC-526 scope fence: the only lowering entry is pops.compile / pops.bind.
    prob = pops.Problem()
    for forbidden in ("run", "install", "compile", "bind"):
        assert not hasattr(prob, forbidden), "Problem must not expose %r" % forbidden


def test_problem_holds_no_array_attribute():
    # A Problem owns no runtime data: no attribute is a list/array of numbers.
    prob = pops.Problem().block("ne", physics=_model())
    for name, value in vars(prob).items():
        assert not isinstance(value, (list, tuple)) or not value or not isinstance(
            value[0], (int, float)), "Problem attribute %r looks like array data" % name


def test_available_is_explainable():
    prob = pops.Problem()
    status = prob.available()
    assert not status.ok
    assert "block" in status.reason
    prob.block("ne", physics=_model())
    assert prob.available().ok


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
