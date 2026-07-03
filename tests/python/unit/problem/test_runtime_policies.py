"""ADC-562: the typed RuntimePolicies bundle isolates runtime concerns from the physics script.

``pops.RuntimePolicies(output=..., checkpoint=..., diagnostics=..., schedules=...)`` groups the
runtime output / checkpoint / diagnostic / schedule policies as ONE typed object; TYPED members
only (a non-descriptor argument RAISES -- no options bag, no string keys). ``problem.runtime(bundle)``
attaches it; the bundle inspects and validates ON ITS OWN and refuses an AMR / MPI / backend-
incompatible member before the runtime is touched. Pure Python; needs only ``import pops``.
"""
import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops.output import (  # noqa: E402
    RuntimePolicies, RuntimePoliciesReport, OutputPolicy, CheckpointPolicy, HDF5)
from pops.diagnostics.measures import Integral, Norm  # noqa: E402
from pops.linalg.norms import L2  # noqa: E402
from pops.time.schedule import every  # noqa: E402


# ---------------------------------------------------------------------------
# Typed-only construction: a non-descriptor argument RAISES.
# ---------------------------------------------------------------------------

def test_typed_members_construct():
    rp = RuntimePolicies(output=OutputPolicy(cadence=every(20)),
                         checkpoint=CheckpointPolicy(cadence=every(100), restartable=True),
                         diagnostics=[Integral(), Norm(L2())], schedules=[every(5)])
    assert rp.options() == {"has_output": True, "has_checkpoint": True,
                            "n_diagnostics": 2, "n_schedules": 1}
    # A single diagnostic / schedule is accepted un-wrapped too.
    assert len(RuntimePolicies(diagnostics=Integral()).diagnostics) == 1


def test_string_output_is_refused():
    with pytest.raises(TypeError, match="typed pops.output policy"):
        RuntimePolicies(output="hdf5")


def test_options_bag_output_is_refused():
    with pytest.raises(TypeError, match="typed pops.output policy"):
        RuntimePolicies(output={"format": "hdf5", "every": 20})


def test_wrong_category_output_is_refused():
    # A CheckpointPolicy passed as output= (right family, wrong slot) is refused.
    with pytest.raises(TypeError, match="output_policy"):
        RuntimePolicies(output=CheckpointPolicy())


def test_non_diagnostic_is_refused():
    with pytest.raises(TypeError, match="pops.diagnostics measures"):
        RuntimePolicies(diagnostics=[OutputPolicy()])


def test_non_schedule_is_refused():
    with pytest.raises(TypeError, match="pops.time.schedule.Schedule"):
        RuntimePolicies(schedules=[object()])


# ---------------------------------------------------------------------------
# Isolated inspect() / requirements() (no physics facade).
# ---------------------------------------------------------------------------

def test_inspect_is_a_typed_report_not_a_dict():
    rp = RuntimePolicies(output=OutputPolicy(cadence=every(20)))
    rep = rp.inspect()
    assert isinstance(rep, RuntimePoliciesReport)
    assert not isinstance(rep, dict)
    d = rep.to_dict()
    assert d["report_type"] == "runtime_policies"
    assert d["output"]["category"] == "output_policy"
    # to_dict() is the explicit bridge and is JSON-ready.
    import json
    assert json.loads(json.dumps(d)) == d


def test_requirements_union_folds_member_requirements():
    rp = RuntimePolicies(output=OutputPolicy(format=HDF5(parallel=True), require_parallel=True))
    assert rp.requirements().to_dict().get("parallel_io") is True


# ---------------------------------------------------------------------------
# validate() refuses an incompatible policy before the runtime (no false positive).
# ---------------------------------------------------------------------------

def test_parallel_only_output_on_serial_context_is_refused():
    rp = RuntimePolicies(output=OutputPolicy(format=HDF5(parallel=True), require_parallel=True))
    report = rp.validate({"parallel": False})
    assert not report.ok
    assert any("parallel_io" in str(i) for i in report.issues)


def test_serial_policy_passes_validation():
    rp = RuntimePolicies(output=OutputPolicy(cadence=every(20)), diagnostics=[Integral()])
    assert rp.validate({}).ok
    assert rp.validate().ok  # no context: never a false positive


def test_unknown_parallel_context_is_not_refused():
    # A parallel-only policy against a context that does NOT declare its parallel state is not
    # rejected (no false positive: gate only on an EXPLICITLY serial backend).
    rp = RuntimePolicies(output=OutputPolicy(format=HDF5(parallel=True), require_parallel=True))
    assert rp.validate({"layout": "uniform"}).ok


# ---------------------------------------------------------------------------
# problem.runtime(bundle): attach + unpack + surface in reports.
# ---------------------------------------------------------------------------

def test_problem_runtime_unpacks_output_and_checkpoint():
    out = OutputPolicy(cadence=every(20))
    chk = CheckpointPolicy(cadence=every(100), restartable=True)
    problem = pops.Problem(name="plasma").runtime(RuntimePolicies(output=out, checkpoint=chk))
    # The bundle's output/checkpoint feed the runtime registry's outputs (run(output_dir=) fires them).
    assert problem._outputs == [out, chk]
    insp = problem._runtime_registry.inspect()
    assert insp["outputs"] == ["OutputPolicy", "CheckpointPolicy"]
    # The bundle report is surfaced in the runtime registry inspection.
    assert insp["policies"]["report_type"] == "runtime_policies"


def test_problem_runtime_refuses_a_non_bundle():
    with pytest.raises(TypeError, match="typed pops.RuntimePolicies bundle"):
        pops.Problem(name="p").runtime(OutputPolicy())


def test_problem_validate_runs_the_bundle_validate():
    # An incompatible policy is refused through problem.validate given a serial-declaring context.
    rp = RuntimePolicies(output=OutputPolicy(format=HDF5(parallel=True), require_parallel=True))
    problem = pops.Problem(name="p").block("ne", physics=type("M", (), {"name": "m"})())
    problem.runtime(rp)
    report = problem.validate_report({"parallel": False})
    assert any(i.family == "runtime_policies" for i in report.issues)


def test_runtime_policies_exported_on_root():
    assert pops.RuntimePolicies is RuntimePolicies


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
