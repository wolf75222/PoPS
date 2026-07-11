"""ADC-659: accepted output/runtime policy fields never disappear or substitute silently."""
from __future__ import annotations

import pytest

from pops import DiagnosticError
from pops.output import HDF5, OutputPolicy, RuntimePolicies
from pops.descriptors_report import RequirementSet
from pops.runtime._amr_output_driver import _write_amr
from pops.time import every


class _Sim:
    def write(self, *args, **kwargs):
        raise AssertionError("a rejected output must not reach a writer")


class _Diagnostic:
    category = "diagnostic_conflict"

    def __init__(self, value):
        self.value = value

    def requirements(self):
        return RequirementSet({"shared": self.value})


def test_free_runtime_schedule_is_rejected_as_unattached():
    report = RuntimePolicies(schedules=[every(3)]).validate({})
    assert not report.ok
    assert any(issue.code == "runtime_policies.unattached_schedule" for issue in report.issues)


def test_requirement_union_rejects_last_writer_wins_conflicts():
    policies = RuntimePolicies(diagnostics=[_Diagnostic(True), _Diagnostic(False)])
    with pytest.raises(ValueError, match="cannot overwrite"):
        policies.requirements()


@pytest.mark.parametrize(
    "policy, message, code",
    [
        (OutputPolicy(format=HDF5()), "NPZ substitution", "runtime.output.hdf5_not_lowered"),
        (OutputPolicy(require_parallel=True), "parallel route",
         "runtime.output.parallel_writer_unavailable"),
    ],
)
def test_amr_output_never_substitutes_a_requested_contract(policy, message, code):
    with pytest.raises(DiagnosticError, match=message) as caught:
        _write_amr(_Sim(), "out", policy, 1, [0])
    assert caught.value.report.code == code
    assert caught.value.report.source == "amr_output"
