"""ADC-550: every typed :class:`pops.Report` round-trips through the inherited ``to_json``.

The nine aggregate inspection reports (the same set fenced by
``tests/python/architecture/test_report_base.py``) all adopt the shared :class:`pops.Report`
base, so none needs its own ``to_json`` / ``_stamp`` boilerplate. This test builds a minimal,
inert instance of each and asserts:

* ``to_json`` (the ONE inherited serialiser) parses back to exactly ``to_dict()`` -- this is the
  byte-identity proof that lets the three redundant ``to_json`` overrides be deleted (they
  re-implemented the base body verbatim, so the base method now runs unchanged);
* ``to_json`` is JSON-stable (``sort_keys``): serialising twice gives the same string;
* ``report_type`` / ``schema_version`` are stable, well-typed drift guards.

Pure Python: constructing a report opens no extension, compiles nothing and binds nothing. The
suite still ``importorskip``s ``pops`` because the report modules live under the package.
"""
import json

import pytest

pytest.importorskip("pops", exc_type=ImportError)

from pops._report import Report  # noqa: E402
from pops.codegen._inspect_compiled_report import CompiledReport  # noqa: E402
from pops.codegen.inspect_compiled import Arguments, MemoryEstimate  # noqa: E402
from pops.codegen.inspect_report import BindReport, RequirementsReport  # noqa: E402
from pops.output.runtime_policies import RuntimePoliciesReport  # noqa: E402
from pops.problem.report_view import ProblemReport  # noqa: E402
from pops.runtime.inspection import RuntimeInspectionReport  # noqa: E402
from pops.time.program_inspect import ProgramReport  # noqa: E402


def _instances():
    """One minimal, inert instance of each of the nine typed reports (no compile / bind)."""
    return [
        ProblemReport({"category": "problem", "name": "p", "blocks": []}),
        ProgramReport(name="t", ops=[], commits=[], hash="deadbeef", histories={},
                      dt_bound=None, scratch={}),
        RuntimePoliciesReport(output=None, checkpoint=None, diagnostics=[], schedules=[],
                              requirements={}),
        Arguments(instances=[], params=[], aux=[], solvers=[], outputs=[],
                  layout_runtime={}, program_name="t"),
        MemoryEstimate(categories={}, cells=0, mesh_shape=(0,), n_cons=0, n_aux=0,
                       scratch_buffers=0, assumptions=[]),
        RequirementsReport(capabilities=[], descriptors=[], constraints={}, unknown=[]),
        BindReport(program_name="t", provided={}, required={}, missing=[]),
        CompiledReport(name="c", backend="production", platform="cpu", layout="system",
                       blocks=[], fields=[], program={}, inputs={}, artifacts={},
                       status={}),
        RuntimeInspectionReport(runtime="System", blocks=[], clock={}, runtime_environment={},
                                capabilities={}, program={}, profile={}, history=[], cache=[],
                                diagnostics={}),
    ]


def test_every_report_json_round_trips_to_its_dict():
    """``Report.to_json`` (inherited) parses back to ``to_dict()`` for every typed report."""
    for report in _instances():
        assert isinstance(report, Report)
        payload = report.to_dict()
        text = report.to_json()
        assert json.loads(text) == payload, (
            "%s: to_json must serialise exactly to_dict()" % type(report).__name__)


def test_report_to_json_is_stable_and_sorted():
    """``to_json`` is deterministic (``sort_keys``): serialising twice is byte-identical."""
    for report in _instances():
        first = report.to_json()
        assert first == report.to_json()
        # sort_keys=True: top-level keys are ordered.
        top = list(json.loads(first).keys())
        assert top == sorted(top), "%s: to_json must be sort_keys-stable" % type(report).__name__


def test_report_type_and_schema_version_are_stable():
    """``report_type`` is a non-empty string and ``schema_version`` an int on every report."""
    for report in _instances():
        assert isinstance(report.report_type, str) and report.report_type
        assert isinstance(report.schema_version, int)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
