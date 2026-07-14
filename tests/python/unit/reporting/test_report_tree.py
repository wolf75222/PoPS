import json
from dataclasses import FrozenInstanceError
import pathlib
import sys
import types

import pytest

# Reporting is stdlib-only and must be testable before the optional native extension is built.  Use
# the real package when available; otherwise install only its package path (not a fake report API).
try:
    import pops
except ImportError:
    for module_name in tuple(sys.modules):
        if module_name == "pops" or module_name.startswith("pops."):
            del sys.modules[module_name]
    pops = types.ModuleType("pops")
    pops.__path__ = [str(pathlib.Path(__file__).resolve().parents[4] / "python" / "pops")]
    sys.modules["pops"] = pops

from pops._report import DiagnosticError, ReportPhase, ReportSeverity, ReportTree
from pops._inspect import explain, inspect as inspect_dict

def _leaf(**overrides):
    values = {
        "phase": "validation",
        "severity": "error",
        "code": "validation.field.invalid",
        "message": "field is invalid",
        "source": "field",
        "notes": ("checked without numerics",),
        "owner": {"kind": "problem", "name": "plasma"},
        "evidence": {"field": "phi", "shape": [8, 8]},
        "actions": ("declare field phi",),
    }
    values.update(overrides)
    return ReportTree(**values)


def test_closed_phase_severity_and_namespaced_code():
    node = _leaf()
    assert node.phase is ReportPhase.VALIDATION
    assert node.severity is ReportSeverity.ERROR

    with pytest.raises(ValueError, match="unknown report phase"):
        _leaf(phase="lowering")
    with pytest.raises(ValueError, match="unknown report severity"):
        _leaf(severity="fatal")
    with pytest.raises(ValueError, match="namespaced"):
        _leaf(code="invalid")


def test_tree_is_deeply_immutable_and_detached_from_inputs():
    owner = {"name": "p", "path": ["case", "p"]}
    evidence = {"field": {"components": ["x", "y"]}}
    node = _leaf(owner=owner, evidence=evidence)
    owner["name"] = "changed"
    evidence["field"]["components"].append("z")

    assert node.to_dict()["owner"]["name"] == "p"
    assert node.to_dict()["evidence"]["field"]["components"] == ["x", "y"]
    with pytest.raises(FrozenInstanceError):
        node.code = "validation.field.changed"
    with pytest.raises(TypeError):
        node.evidence["field"] = {}
    with pytest.raises(TypeError):
        node.evidence["field"]["components"][0] = "z"


def test_recursive_ok_functional_composition_and_strict_exception():
    root = ReportTree(
        phase="validation", severity="info", code="validation.problem.report",
        children=(ReportTree(
            phase="validation", severity="warning", code="validation.field.deprecated"),),
    )
    assert root.ok
    failed = root.with_child(_leaf())
    assert root.ok and not failed.ok
    assert [issue.code for issue in failed.issues] == [
        "validation.field.deprecated", "validation.field.invalid"]
    with pytest.raises(DiagnosticError) as caught:
        failed.raise_if_error()
    assert caught.value.report is failed
    with pytest.raises(ValueError, match="containing an error"):
        DiagnosticError(root)


def test_error_and_extend_are_functional_tree_composition():
    root = ReportTree(
        phase="validation", severity="info", code="validation.problem.report",
        owner={"kind": "case", "name": "p"},
    )
    failed = root.error(
        "field", "invalid", "field is invalid", context={"field": "phi"},
        evidence={"components": 1}, alternatives=("remove phi",),
        actions=("declare phi",), notes=("metadata-only check",),
    )
    assert root.ok and root.children == ()
    assert not failed.ok and failed is not root
    issue = failed.children[0]
    assert issue.code == "field.invalid"
    assert issue.source == "field"
    assert issue.evidence == {"components": 1, "field": "phi"}
    assert issue.actions == ("remove phi", "declare phi")
    assert issue.owner == root.owner

    underscored = root.error("elliptic_solver", "not_wired", "not wired")
    assert underscored.children[0].code == "elliptic_solver.not_wired"

    other = ReportTree(
        phase="validation", severity="warning", code="validation.output.deprecated")
    combined = failed.extend(other)
    assert combined.children == failed.children + (other,)
    assert failed.children == (issue,)
    with pytest.raises(TypeError, match="expects a ReportTree"):
        root.extend(None)


def test_dict_and_json_round_trip_are_deterministic(tmp_path):
    tree = ReportTree(
        phase="validation", severity="info", code="validation.problem.report",
        evidence={"z": 2, "a": 1}, children=(_leaf(),),
    )
    payload = tree.to_dict()
    rebuilt = ReportTree.from_dict(payload)
    assert rebuilt == tree
    assert rebuilt.to_dict() == payload
    assert ReportTree.from_json(tree.to_json()) == tree
    assert json.loads(tree.to_json(indent=None)) == payload

    path = tmp_path / "report.json"
    assert tree.to_json(path) == path
    assert ReportTree.from_json(path) == tree

    drifted = dict(payload, ok=True)
    with pytest.raises(ValueError, match="verdict"):
        ReportTree.from_dict(drifted)


def test_owner_identity_is_projected_without_retaining_live_owner():
    class Owner:
        def __init__(self):
            self.data = {"kind": "case", "name": "p"}

        def canonical(self):
            return self

        def to_data(self):
            return self.data

    owner = Owner()
    tree = _leaf(owner=owner)
    owner.data["name"] = "changed"
    assert tree.to_dict()["owner"] == {"kind": "case", "name": "p"}


def test_pops_inspect_is_dict_bridge_and_explain_remains_typed():
    tree = _leaf()
    assert inspect_dict(tree) == tree.to_dict()
    assert explain(tree) is tree

    class Explained:
        def explain(self):
            return tree

    assert explain(Explained()) is tree

    class Inspected:
        name = "demo"

        def inspect(self):
            return {"value": 3}

    generic = explain(Inspected())
    assert isinstance(generic, ReportTree)
    assert generic.phase is ReportPhase.INSPECTION
    assert generic.evidence["inspection"]["value"] == 3


def test_json_evidence_rejects_live_or_non_json_values():
    with pytest.raises(TypeError, match="non-JSON"):
        _leaf(evidence={"value": object()})
    with pytest.raises(ValueError, match="non-finite"):
        _leaf(evidence={"value": float("nan")})
