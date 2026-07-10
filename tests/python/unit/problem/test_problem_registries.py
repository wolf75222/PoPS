"""ADC-553: the typed internal registries of a Problem are independently inspectable/validatable.

Each registry (blocks / fields / time / params / runtime policies / constraints) exposes add / get /
names / __iter__ / inspect / validate and reports structured per-family errors. Problem.validate()
aggregates the child reports into one ProblemValidationReport whose by_family() lists the errors per
subsystem.

Pure Python; needs only `import pops`.
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.problem.registries import (  # noqa: E402
    BlockRegistry, ConstraintRegistry, FieldRegistry, ParamRegistry,
    RuntimePolicyRegistry, TimeRegistry)
from pops.problem.report import ProblemValidationReport  # noqa: E402
from pops.model import OwnerPath  # noqa: E402


_OWNER = OwnerPath("problem", "registry-tests")


class _StubModel:
    def __init__(self, name="stub"):
        self.name = name


def test_block_registry_add_get_names_and_duplicate():
    reg = BlockRegistry(owner=_OWNER)
    handle = reg.add("ne", _StubModel(), spatial=None)
    assert handle.name == "ne"
    assert handle.qualified_id == (
        "pops.handle.v1::problem/registry-tests::block::ne")
    assert reg.names() == ["ne"]
    assert reg.get("ne")["model"].name == "stub"
    with pytest.raises(ValueError):
        reg.add("ne", _StubModel())
    assert list(reg) == ["ne"]
    assert "ne" in reg


def test_block_registry_validate_reports_no_block():
    report = BlockRegistry(owner=_OWNER).validate()
    assert isinstance(report, ProblemValidationReport)
    assert not report.ok
    assert any(i.code == "no_block" for i in report)


def test_field_registry_type_checks_and_names():
    reg = FieldRegistry(owner=_OWNER)
    with pytest.raises(TypeError):
        reg.add("not a field")
    from pops.fields import PoissonProblem
    from pops.math import laplacian
    from pops.solvers import GeometricMG
    fp = PoissonProblem(name="phi", unknown="phi",
                        equation=(-laplacian("phi") == "charge_density"),
                        solver=GeometricMG())
    handle = reg.add(fp)
    assert handle.qualified_id == (
        "pops.handle.v1::problem/registry-tests::field::phi")
    assert reg.names() == ["phi"]
    assert reg.solvers()  # phi has a solver


def test_time_registry_single_slot():
    reg = TimeRegistry()
    assert reg.program is None
    reg.set(object())
    assert reg.program is not None
    assert reg.validate().ok


def test_param_registry_rejects_kind_string():
    reg = ParamRegistry()
    reg.add("alpha", 1.0)
    assert reg.get("alpha") == {"default": 1.0, "kind": "const"}
    with pytest.raises(TypeError, match="kind="):
        reg.add("beta", 1.0, kind="const")


def test_registries_never_stringify_author_identity_objects():
    block = BlockRegistry(owner=_OWNER)
    params = ParamRegistry()
    runtime = RuntimePolicyRegistry()

    with pytest.raises(TypeError, match="block name"):
        block.add(object(), _StubModel())
    with pytest.raises(TypeError, match="parameter name"):
        params.add(object(), 1.0)
    with pytest.raises(TypeError, match="aux name"):
        runtime.add_aux(object(), 1.0)
    assert block.names() == []
    assert params.names() == []
    assert runtime.aux == {}


def test_runtime_policy_registry_refuses_bad_output():
    reg = RuntimePolicyRegistry()

    class _NotAPolicy:
        name = "nope"
    reg.add_output(_NotAPolicy())
    report = reg.validate()
    assert not report.ok
    assert any(i.code == "bad_output_policy" for i in report)


def test_constraint_registry_records_refinement_layout_free():
    reg = ConstraintRegistry()
    from pops.mesh.amr import RegridEvery
    reg.set_refinement(regrid=RegridEvery(20))
    assert "regrid" in reg.refinement
    # No layout at assembly, so nothing to reject here.
    assert reg.validate().ok


def test_problem_validate_aggregates_by_family():
    # A Problem with a bad output AND a name collision reports BOTH families in one pass.
    from pops.fields import PoissonProblem
    from pops.math import laplacian
    from pops.solvers import GeometricMG

    class _NotAPolicy:
        name = "nope"
    fp = PoissonProblem(name="ne", unknown="ne",
                        equation=(-laplacian("ne") == "charge_density"),
                        solver=GeometricMG())
    prob = (pops.Problem().block("ne", physics=_StubModel())
            .field(fp)              # collides with block name "ne"
            .output(_NotAPolicy()))
    report = prob.validate_report()
    families = report.by_family()
    assert "runtime" in families and "field" in families
    assert not report.ok


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
