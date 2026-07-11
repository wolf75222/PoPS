"""ADC-659: behavior-bearing Problem fields lower exactly or reject before compilation."""
from __future__ import annotations

import pytest

import pops
from pops.diagnostics import Integral
from pops.model import Module
from pops.output import RuntimePolicies
from pops.time import Program, every


def _program(name):
    return Program(name)


def test_unattached_runtime_schedule_is_not_a_successful_problem():
    problem = pops.Problem(name="free-schedule")
    problem.add_block("fluid", Module("m"))
    problem.runtime(RuntimePolicies(schedules=[every(2)]))
    report = problem.validate_report()
    assert any(issue.code == "runtime_policies.unattached_schedule" for issue in report.issues)


def test_block_time_and_diagnostics_never_drop_from_resolved_plan():
    problem = pops.Problem(name="block-fields")
    problem.add_block(
        "fluid", Module("m"), time=_program("local"), diagnostics=(Integral(),))
    codes = {issue.code for issue in problem.validate_report().issues}
    assert {"block.unlowered_block_time", "block.unlowered_block_diagnostics"} <= codes


def test_compile_time_cannot_overwrite_problem_time(monkeypatch):
    from pops.codegen import orchestration

    problem = pops.Problem(name="time-authority")
    problem.add_block("fluid", Module("m"))
    problem.time(_program("owned"))
    layout = object()
    monkeypatch.setattr(orchestration, "_resolve_layout", lambda *_: layout)
    monkeypatch.setattr(orchestration, "_validate_layout_for_compile", lambda *_: None)
    with pytest.raises(ValueError, match="competing semantic authorities"):
        orchestration.compile(problem, layout=layout, time=_program("override"))
