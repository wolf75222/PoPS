"""ADC-553: the typed internal registries of a Problem are independently inspectable/validatable.

Each registry (blocks / fields / time / params / runtime policies / constraints) exposes add / get /
names / __iter__ / inspect / validate and reports structured per-source errors. Problem.validate()
aggregates the child trees into one immutable ReportTree whose by_source() lists errors per subsystem.

Pure Python; needs only `import pops`.
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.problem.registries import (  # noqa: E402
    BlockRegistry, ConstraintRegistry, FieldRegistry, ParamRegistry,
    RuntimePolicyRegistry, TimeRegistry)
from pops import ReportTree  # noqa: E402
from pops.model import OwnerKind, OwnerPath  # noqa: E402
from pops.params import ConstParam  # noqa: E402


_OWNER = OwnerPath.fresh(OwnerKind.CASE, "registry-tests")


class _StubModel:
    def __init__(self, name="stub"):
        self.name = name
        self.owner_path = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, name)


def test_block_registry_add_get_names_and_duplicate():
    reg = BlockRegistry(owner=_OWNER)
    handle = reg.add("ne", _StubModel(), spatial=None)
    assert handle.name == "ne"
    assert handle.owner_path == _OWNER
    assert handle.kind == "block"
    assert reg.names() == ["ne"]
    assert reg.get("ne")["model"].name == "stub"
    with pytest.raises(ValueError):
        reg.add("ne", _StubModel())
    assert list(reg) == ["ne"]
    assert "ne" in reg


def test_block_registry_validate_reports_no_block():
    report = BlockRegistry(owner=_OWNER).validate()
    assert isinstance(report, ReportTree)
    assert not report.ok
    assert any(i.code == "block.no_block" for i in report.issues)


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
    assert handle.owner_path == _OWNER
    assert handle.kind == "field"
    assert reg.names() == ["phi"]
    assert reg.solvers()  # phi has a solver


def test_time_registry_single_slot():
    reg = TimeRegistry()
    assert reg.program is None
    program = object()
    reg.set(program)
    assert reg.program is program
    with pytest.raises(ValueError, match="already declared"):
        reg.set(program)
    assert reg.program is program
    assert reg.validate().ok


def test_param_registry_rejects_kind_string():
    reg = ParamRegistry(owner=_OWNER)
    alpha = ConstParam("alpha", 1.0)
    handle = reg.add(alpha)
    assert reg.get(handle) is alpha
    assert reg.get("alpha") is alpha
    with pytest.raises(TypeError, match="kind"):
        ConstParam("beta", 1.0, kind="const")


def test_param_registry_rejects_identical_and_incompatible_redeclarations():
    reg = ParamRegistry(owner=_OWNER)
    alpha = ConstParam("alpha", 1.0)
    reg.add(alpha)
    with pytest.raises(ValueError, match="already declared"):
        reg.add(alpha)
    with pytest.raises(ValueError, match="already declared"):
        reg.add(ConstParam("alpha", 2.0))
    assert reg.get("alpha") is alpha


def test_registries_never_stringify_author_identity_objects():
    block = BlockRegistry(owner=_OWNER)
    params = ParamRegistry(owner=_OWNER)
    runtime = RuntimePolicyRegistry()

    with pytest.raises(TypeError, match="block name"):
        block.add(object(), _StubModel())
    with pytest.raises(TypeError, match="RuntimeParam, ConstParam or DerivedParam"):
        params.add(object())
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
    assert any(i.code == "runtime.bad_output_policy" for i in report.issues)


def test_runtime_registry_declarations_are_register_once():
    from pops.diagnostics.measures import Integral
    from pops.output import OutputPolicy
    from pops.time.schedule import every

    reg = RuntimePolicyRegistry()
    reg.add_aux("B_z", 1.0)
    with pytest.raises(ValueError, match="already declared"):
        reg.add_aux("B_z", 1.0)

    output = OutputPolicy(cadence=every(5))
    reg.add_output(output)
    with pytest.raises(ValueError, match="already registered"):
        reg.add_output(OutputPolicy(cadence=every(5)))

    diagnostic = Integral()
    reg.add_diagnostic(diagnostic)
    with pytest.raises(ValueError, match="already registered"):
        reg.add_diagnostic(Integral())
    assert reg.aux == {"B_z": 1.0}
    assert reg.outputs == [output]
    assert reg.diagnostics == [diagnostic]


def test_runtime_policy_bundle_is_register_once_and_collision_is_transactional():
    from pops.diagnostics.measures import Integral
    from pops.output import OutputPolicy, RuntimePolicies
    from pops.time.schedule import every

    reg = RuntimePolicyRegistry()
    existing = OutputPolicy(cadence=every(5))
    reg.add_output(existing)
    bundle = RuntimePolicies(
        output=OutputPolicy(cadence=every(5)), diagnostics=[Integral()])
    with pytest.raises(ValueError, match="repeats an output declaration"):
        reg.set_policies(bundle)
    assert reg.bundle_declared is False
    assert reg.outputs == [existing]
    assert reg.diagnostics == []

    accepted = RuntimePolicies(
        output=OutputPolicy(cadence=every(10)), diagnostics=[Integral()])
    reg.set_policies(accepted)
    with pytest.raises(ValueError, match="already declared"):
        reg.set_policies(accepted)
    assert reg.bundle_declared is True
    assert not hasattr(reg, "_policies")


def test_runtime_policy_bundle_rejects_internal_duplicate_diagnostics():
    from pops.diagnostics.measures import Integral
    from pops.output import RuntimePolicies

    reg = RuntimePolicyRegistry()
    with pytest.raises(ValueError, match="duplicate diagnostic"):
        reg.set_policies(RuntimePolicies(diagnostics=[Integral(), Integral()]))
    assert reg.bundle_declared is False
    assert reg.diagnostics == []


def test_constraint_registry_records_refinement_layout_free():
    reg = ConstraintRegistry()
    from pops.mesh.amr import RegridEvery
    reg.set_refinement(regrid=RegridEvery(20))
    assert "regrid" in reg.refinement
    # No layout at assembly, so nothing to reject here.
    assert reg.validate().ok


def test_constraint_registry_rejects_duplicate_criterion_without_partial_update():
    reg = ConstraintRegistry()
    from pops.mesh.amr import RegridEvery
    original = RegridEvery(20)
    reg.set_refinement(regrid=original)
    with pytest.raises(ValueError, match="already declared for: regrid"):
        reg.set_refinement(refine=object(), regrid=RegridEvery(10))
    assert reg.refinement == {"regrid": original}


def test_cross_family_homonyms_remain_typed_and_do_not_collide():
    # A block and a field may share a display name: kind is part of Handle identity.
    from pops.fields import PoissonProblem
    from pops.math import laplacian
    from pops.solvers import GeometricMG

    class _NotAPolicy:
        name = "nope"
    fp = PoissonProblem(name="ne", unknown="ne",
                        equation=(-laplacian("ne") == "charge_density"),
                        solver=GeometricMG())
    prob = (pops.Problem().block("ne", physics=_StubModel())
            .field(fp)
            .output(_NotAPolicy()))
    report = prob.validate_report()
    sources = report.by_source()
    assert "runtime" in sources and "field" not in sources
    assert prob.blocks()["ne"] != prob.fields()["ne"]
    assert not report.ok


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
