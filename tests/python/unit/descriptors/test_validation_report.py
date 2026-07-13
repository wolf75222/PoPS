"""ADC-527 / ADC-625 / ADC-659 typed descriptor results and reports.

ADC-625 makes them the ONE final form: TYPED objects, NOT dict subclasses. The only mapping
bridge is ``to_dict()``; the typed accessors (``add`` / ``check`` / ``supports`` and the
``LoweredDescriptor`` attributes) are the interface.

Pure Python; needs only `import pops`.
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops._report import DiagnosticError, ReportTree  # noqa: E402
from pops.descriptors import (  # noqa: E402
    CapabilitySet, LoweredDescriptor, Requirement, RequirementSet)


def test_requirement_set_is_typed_not_a_dict_subclass():
    r = RequirementSet({"time_scheme": True, "elliptic_solve": True})
    assert not isinstance(r, dict)  # ADC-625: a typed object, not a dict subclass
    assert r.to_dict()["time_scheme"] is True
    assert r.to_dict().get("missing", "d") == "d"
    assert set(r.to_dict()) == {"time_scheme", "elliptic_solve"}
    assert r.to_dict() == {"time_scheme": True, "elliptic_solve": True}


def test_requirement_set_from_iterable_and_add_chains():
    r = RequirementSet([Requirement("mpi"), Requirement("gpu", value=False)])
    assert r.to_dict()["mpi"] is True and r.to_dict()["gpu"] is False
    assert r.add("amr").add("uniform") is r
    assert "amr" in r.to_dict() and "uniform" in r.to_dict()


def test_requirement_set_check_reports_unsatisfied():
    r = RequirementSet({"time_scheme": True})
    report = r.check({})            # empty context -> unsatisfied
    assert not report.ok
    assert any(i.code == "validation.requirement.unsatisfied" for i in report.issues)
    assert r.check({"time_scheme": True}).ok


def test_capability_set_supports_and_from_dict():
    c = CapabilitySet.from_dict({"supports_amr": True, "supports_gpu": False})
    assert not isinstance(c, dict)  # ADC-625: a typed object, not a dict subclass
    assert c.supports("amr") is True
    assert c.supports("gpu") is False
    assert c.supports("mpi") is False       # absent -> False, never raises
    assert c.to_dict() == {"supports_amr": True, "supports_gpu": False}


def test_lowered_descriptor_exposes_attributes_and_to_dict():
    ld = LoweredDescriptor(name="HLL", category="riemann", native_id="pops::HLLFlux",
                           options={"order": 1})
    assert not isinstance(ld, dict)  # ADC-625: a typed object, not a dict subclass
    assert ld.name == "HLL" and ld.native_id == "pops::HLLFlux"
    assert ld.to_dict()["name"] == "HLL" and ld.to_dict()["native_id"] == "pops::HLLFlux"
    assert ld.to_dict()["category"] == "riemann"


def test_report_tree_composes_by_source_without_mutation():
    report = ReportTree(
        phase="validation", severity="info", code="validation.descriptor.report")
    original = report
    report = report.error("block", "no_model", "block ne has no model",
                          context={"block": "ne"})
    report = report.error("field", "unbound", "field psi has no solver")
    assert not report.ok
    sources = report.by_source()
    assert set(sources) == {"block", "field"}
    assert len(report.issues) == 2
    assert original.ok and original.children == ()


def test_report_tree_raise_if_error_is_fail_loud_and_structured():
    ok = ReportTree(
        phase="validation", severity="info", code="validation.descriptor.report")
    ok.raise_if_error()  # no raise on a clean report
    assert ok.ok
    bad = ok.error("time", "incompatible", "Program has no step")
    with pytest.raises(DiagnosticError, match="incompatible") as caught:
        bad.raise_if_error()
    assert caught.value.report is bad


def test_report_error_carries_alternatives_as_actions():
    root = ReportTree(
        phase="validation", severity="info", code="validation.descriptor.report")
    report = root.error(
        "descriptor", "unavailable", "no native symbol",
        alternatives=["pops.inspect(descriptor)"])
    issue = report.issues[0]
    assert issue.actions == ("pops.inspect(descriptor)",)
    assert "no native symbol" in str(issue)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
