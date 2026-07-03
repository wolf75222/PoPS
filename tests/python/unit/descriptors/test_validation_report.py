"""ADC-527: the typed result objects (RequirementSet / CapabilitySet / LoweredDescriptor /
ValidationReport) behave per the frozen contract and stay Mapping-compatible.

Pure Python; needs only `import pops`.
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.descriptors import (  # noqa: E402
    CapabilitySet, LoweredDescriptor, Requirement, RequirementSet, ValidationIssue,
    ValidationReport)


def test_requirement_set_is_mapping_compatible():
    r = RequirementSet({"time_scheme": True, "elliptic_solve": True})
    assert isinstance(r, dict)  # back-compat: it IS a mapping
    assert r["time_scheme"] is True
    assert r.get("missing", "d") == "d"
    assert set(r) == {"time_scheme", "elliptic_solve"}
    assert r.to_dict() == {"time_scheme": True, "elliptic_solve": True}


def test_requirement_set_from_iterable_and_add_chains():
    r = RequirementSet([Requirement("mpi"), Requirement("gpu", value=False)])
    assert r["mpi"] is True and r["gpu"] is False
    assert r.add("amr").add("uniform") is r
    assert "amr" in r and "uniform" in r


def test_requirement_set_check_reports_unsatisfied():
    r = RequirementSet({"time_scheme": True})
    report = r.check({})            # empty context -> unsatisfied
    assert not report.ok
    assert any(i.code == "unsatisfied" for i in report)
    assert r.check({"time_scheme": True}).ok


def test_capability_set_supports_and_from_dict():
    c = CapabilitySet.from_dict({"supports_amr": True, "supports_gpu": False})
    assert isinstance(c, dict)
    assert c.supports("amr") is True
    assert c.supports("gpu") is False
    assert c.supports("mpi") is False       # absent -> False, never raises


def test_lowered_descriptor_is_dict_superset_and_inert():
    ld = LoweredDescriptor(name="HLL", category="riemann", native_id="pops::HLLFlux",
                           options={"order": 1})
    assert isinstance(ld, dict)
    assert ld["name"] == "HLL" and ld["native_id"] == "pops::HLLFlux"
    assert ld.native_id == "pops::HLLFlux"
    assert ld.to_dict()["category"] == "riemann"


def test_validation_report_accumulates_by_family():
    report = ValidationReport()
    report.error("block", "no_model", "block ne has no model", context={"block": "ne"})
    report.error("field", "unbound", "field psi has no solver")
    assert not report.ok
    families = report.by_family()
    assert set(families) == {"block", "field"}
    assert len(report) == 2


def test_validation_report_raise_if_error_is_fail_loud():
    ok = ValidationReport()
    ok.raise_if_error()  # no raise on a clean report
    assert ok.ok
    bad = ValidationReport().error("time", "incompatible", "Program has no step")
    with pytest.raises(ValueError, match="incompatible"):
        bad.raise_if_error()


def test_validation_issue_carries_alternatives():
    issue = ValidationIssue(family="descriptor", code="unavailable", message="no native symbol",
                            alternatives=["pops.inspect_capabilities()"])
    assert issue.alternatives == ["pops.inspect_capabilities()"]
    assert "alternatives" in str(issue)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
