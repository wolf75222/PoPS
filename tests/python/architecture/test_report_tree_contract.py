"""ADC-659 source fences for the single immutable report-tree authority."""
import ast
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = ROOT / "python" / "pops"


def _tree(relative):
    path = POPS / relative
    return ast.parse(path.read_text(), str(path))


def test_report_tree_is_frozen_and_closed_vocabularies_are_exact():
    tree = _tree("_report.py")
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    report = classes["ReportTree"]
    decorators = [node for node in report.decorator_list if isinstance(node, ast.Call)]
    dataclass = next(node for node in decorators if getattr(node.func, "id", None) == "dataclass")
    keywords = {item.arg: item.value for item in dataclass.keywords}
    assert isinstance(keywords["frozen"], ast.Constant) and keywords["frozen"].value is True

    def enum_values(name):
        return {
            node.value.value for node in classes[name].body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        }

    assert enum_values("ReportPhase") == {
        "authoring", "validation", "compile", "bind", "runtime", "inspection"}
    assert enum_values("ReportSeverity") == {"trace", "info", "warning", "error"}


def test_no_parallel_validation_issue_or_report_classes_remain():
    for relative in ("descriptors_report.py", "problem/report.py"):
        classes = {
            node.name for node in _tree(relative).body if isinstance(node, ast.ClassDef)}
        assert "ValidationIssue" not in classes
        assert "ValidationReport" not in classes
        source = (POPS / relative).read_text()
        assert "ProblemValidationReport =" not in source
        assert "ProblemValidationIssue =" not in source


def test_root_exports_explain_but_keeps_report_tree_as_internal_authority():
    root_source = (POPS / "__init__.py").read_text()
    root_tree = ast.parse(root_source)
    exported = set()
    for node in ast.walk(root_tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets) and isinstance(node.value, (ast.List, ast.Tuple)):
            exported.update(
                item.value for item in node.value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str))

    assert "from ._inspect import explain, inspect" in root_source
    assert {"explain", "inspect"} <= exported
    assert {"ReportTree", "ReportPhase", "ReportSeverity", "DiagnosticError"}.isdisjoint(exported)
    assert "from ._report import" not in root_source

    inspect_source = (POPS / "_inspect.py").read_text()
    assert "from pops._report import ReportTree" in inspect_source
    assert "def explain(" in inspect_source
