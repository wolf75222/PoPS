"""Make every skipped, xfailed or xpassed M2 proof fail its pytest session."""

from __future__ import annotations

import pytest


_FORBIDDEN_OUTCOMES: list[tuple[str, str]] = []


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    _FORBIDDEN_OUTCOMES.clear()


def _record(report: pytest.TestReport | pytest.CollectReport) -> None:
    was_xfail = bool(getattr(report, "wasxfail", False))
    if report.skipped:
        outcome = "xfail" if was_xfail else "skip"
    elif report.passed and was_xfail:
        outcome = "xpass"
    else:
        return
    _FORBIDDEN_OUTCOMES.append((outcome, report.nodeid))


def pytest_collectreport(report: pytest.CollectReport) -> None:
    _record(report)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    _record(report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    if _FORBIDDEN_OUTCOMES:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if not _FORBIDDEN_OUTCOMES:
        return
    terminalreporter.section("M2 mandatory-proof violations", red=True, bold=True)
    for outcome, nodeid in _FORBIDDEN_OUTCOMES:
        terminalreporter.write_line("%s: %s" % (outcome, nodeid), red=True)
