"""ADC-564: the typed inspection reports share ONE base and are NEVER dict subclasses.

Source-and-import guards on the report family:

* ``python/pops/_report.py`` defines the shared :class:`Report` base (report_type / schema_version /
  to_dict / to_json / __str__), and it is NOT a ``dict`` subclass;
* every aggregate inspection report (Problem / Program / compiled / requirements / bind / arguments /
  memory-estimate / runtime-inspection / runtime-policies) adopts that base and is NOT a dict
  subclass -- the only mapping bridge is ``to_dict()``;
* a report builder imports no ``_pops`` at module scope, so building a report triggers no native
  load / compile / bind (a report is inert).

The dict-subclass guard is source-only (ast); the base-adoption guard imports ``pops`` (skipped when
the extension is unavailable, like every runtime-tier test).
"""
import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"

# The report modules and the classes each must define WITHOUT subclassing dict.
_REPORT_CLASSES = {
    "_report.py": ("Report",),
    "problem/report_view.py": ("ProblemReport",),
    "time/program_inspect.py": ("ProgramReport",),
    "output/runtime_policies.py": ("RuntimePoliciesReport",),
    "codegen/inspect_compiled.py": ("Arguments", "MemoryEstimate"),
    "codegen/inspect_report.py": ("RequirementsReport", "BindReport"),
    "codegen/_inspect_compiled_report.py": ("CompiledReport",),
    "runtime/inspection.py": ("RuntimeInspectionReport",),
}


def _classes(path):
    tree = ast.parse(path.read_text(), str(path))
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def test_report_base_exists_and_is_not_a_dict_subclass():
    path = POPS / "_report.py"
    assert path.exists(), "python/pops/_report.py must define the shared Report base (ADC-564)"
    report = _classes(path)["Report"]
    bases = [b.id for b in report.bases if isinstance(b, ast.Name)]
    assert "dict" not in bases, "Report must NOT subclass dict (ADC-564): to_dict() is the bridge"


def test_no_report_is_a_dict_subclass():
    for rel, names in _REPORT_CLASSES.items():
        classes = _classes(POPS / rel)
        for name in names:
            assert name in classes, "%s must define %r" % (rel, name)
            bases = [b.id for b in classes[name].bases if isinstance(b, ast.Name)]
            assert "dict" not in bases, (
                "%s.%s must NOT subclass dict (ADC-564): the only mapping bridge is to_dict()"
                % (rel, name))


def test_report_module_imports_no_pops_extension():
    # A report is inert: building one must not load the native extension. The base module imports
    # only the stdlib.
    src = (POPS / "_report.py").read_text()
    assert "_pops" not in src, "pops._report must not import the native extension (a report is inert)"


def test_reports_adopt_the_base_and_are_not_dicts():
    pytest.importorskip("pops", exc_type=ImportError)
    from pops._report import Report
    from pops.problem.report_view import ProblemReport
    from pops.time.program_inspect import ProgramReport
    from pops.output.runtime_policies import RuntimePoliciesReport
    from pops.codegen.inspect_compiled import Arguments, MemoryEstimate
    from pops.codegen.inspect_report import RequirementsReport, BindReport
    from pops.codegen._inspect_compiled_report import CompiledReport
    from pops.runtime.inspection import RuntimeInspectionReport

    for cls in (ProblemReport, ProgramReport, RuntimePoliciesReport, Arguments, MemoryEstimate,
                RequirementsReport, BindReport, CompiledReport, RuntimeInspectionReport):
        assert issubclass(cls, Report), "%s must adopt the pops.Report base (ADC-564)" % cls.__name__
        assert not issubclass(cls, dict), "%s must not subclass dict" % cls.__name__
        assert isinstance(cls.report_type, str) and cls.report_type


def test_problem_and_program_inspect_return_typed_reports_and_do_not_compile():
    pops = pytest.importorskip("pops", exc_type=ImportError)
    from pops._report import Report
    from pops.model import Module

    # Problem.inspect() -> a typed report; building it triggers no compile / bind (it reads metadata).
    model = Module("m")
    state_space = model.state_space("U", ("u",))
    state = model.state_handle(state_space)
    prob = pops.Problem(name="p")
    block = prob.add_block("ne", model)
    prob_report = prob.inspect()
    assert isinstance(prob_report, Report) and not isinstance(prob_report, dict)
    assert prob_report.category == "problem"
    # pops.inspect(obj) is the explicit dict bridge over the report's to_dict().
    assert pops.inspect(prob) == prob_report.to_dict()

    prog = pops.time.Program("t")
    prog.state(block, state)
    prog_report = prog.inspect()
    assert isinstance(prog_report, Report) and prog_report.report_type == "program"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
