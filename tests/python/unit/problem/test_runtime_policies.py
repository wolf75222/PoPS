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
from pops.model import Module  # noqa: E402
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


def test_output_policy_rejects_untyped_diagnostics():
    with pytest.raises(TypeError, match="typed pops.diagnostics measures"):
        OutputPolicy(diagnostics=["mass"])


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
    # RuntimePolicies is an input bundle; the registry owns each flattened declaration once.
    assert insp["bundle_declared"] is True
    assert "policies" not in insp
    assert not hasattr(problem._runtime_registry, "_policies")


def test_problem_runtime_refuses_a_non_bundle():
    with pytest.raises(TypeError, match="typed pops.RuntimePolicies bundle"):
        pops.Problem(name="p").runtime(OutputPolicy())


def test_problem_validate_runs_the_bundle_validate():
    # An incompatible policy is refused through problem.validate given a serial-declaring context.
    rp = RuntimePolicies(output=OutputPolicy(format=HDF5(parallel=True), require_parallel=True))
    problem = pops.Problem(name="p").block("ne", physics=Module("m"))
    problem.runtime(rp)
    report = problem.validate_report({"parallel": False})
    assert any(i.source == "runtime_policies" for i in report.issues)


def test_runtime_bundle_is_flattened_once_in_problem_snapshot():
    output = OutputPolicy(cadence=every(20))
    diagnostic = Integral()
    schedule = every(5)
    bundle = RuntimePolicies(
        output=output, diagnostics=[diagnostic], schedules=[schedule])
    problem = pops.Problem(name="runtime-snapshot")
    problem.add_block("fluid", Module("runtime-model"))
    problem.runtime(bundle)

    payload = problem.freeze().to_dict()
    assert "runtime_policies" not in payload
    assert len(payload["outputs"]) == 1
    assert len(payload["diagnostics"]) == 1
    assert len(payload["schedules"]) == 1
    assert not hasattr(problem._runtime_registry, "_policies")


def test_problem_validates_output_field_ownership_before_lowering():
    module = Module("transport-output")
    state = module.state_space("U", components=("rho",))
    state_ref = module.state_handle(state)

    ambiguous = pops.Problem(name="ambiguous-output")
    block_a = ambiguous.add_block("a", module)
    block_b = ambiguous.add_block("b", module)
    ambiguous.output(OutputPolicy(fields=[state_ref]))
    report = ambiguous.validate_report()
    issue = next(
        item for item in report.issues
        if item.code == "runtime.ambiguous_declaration_reference")
    assert str(block_a.instance_owner_path) in issue.message
    assert str(block_b.instance_owner_path) in issue.message

    resolved = pops.Problem(name="resolved-output")
    block = resolved.add_block("a", module)
    resolved.output(OutputPolicy(fields=[block[state_ref]]))
    assert resolved.validate_report().ok


def test_runtime_registry_rejects_non_writable_output_handle_kind():
    module = Module("bad-output-kind")
    state = module.state_space("U", components=("rho",))
    module.operator(
        "rhs", signature=(state,) >> pops.model.Rate(state),
        kind="local_rate", expr="rhs")
    operator = module.operator_handle("rhs")

    # Mutate after construction to exercise the registry trust boundary too; the
    # OutputPolicy constructor independently rejects this public misuse.
    policy = OutputPolicy()
    policy.fields = [operator]
    problem = pops.Problem(name="bad-output-kind")
    problem.add_block("fluid", module)
    problem.output(policy)
    issue = next(
        item for item in problem.validate_report().issues
        if item.code == "runtime.invalid_output_field_kind")
    assert "local_rate" in issue.message


def test_problem_validates_diagnostic_block_ownership_before_lowering():
    module = Module("transport-diagnostic")
    owner = pops.Problem(name="diagnostic-owner")
    local_block = owner.add_block("local", module)
    owner.runtime(RuntimePolicies(diagnostics=[Integral(block=local_block)]))
    assert owner.validate_report().ok

    foreign = pops.Problem(name="diagnostic-foreign")
    foreign.add_block("other", module)
    foreign.runtime(RuntimePolicies(diagnostics=[Integral(block=local_block)]))
    report = foreign.validate_report()
    issue = next(
        item for item in report.issues
        if item.code == "runtime.invalid_declaration_reference")
    assert "not registered by this case" in issue.message


def test_compiled_runtime_snapshot_detaches_and_canonicalizes_references():
    from pops.codegen.orchestration import _capture_runtime_declarations
    from pops.runtime._output_driver import _field_names

    module = Module("snapshot-output")
    state = module.state_space("U", components=("rho",))
    state_ref = module.state_handle(state)
    problem = pops.Problem(name="snapshot-output")
    block = problem.add_block("transport", module)
    measure = Integral(block=block)
    policy = OutputPolicy(fields=[state_ref], diagnostics=[measure])
    problem.output(policy)
    problem.runtime(RuntimePolicies(diagnostics=[Norm(L2(), block=block)]))

    outputs, diagnostics = _capture_runtime_declarations(problem)
    captured = outputs[0]
    assert captured is not policy
    assert captured.fields[0].is_resolved
    assert captured.fields[0].block_ref.local_id == "transport"
    assert captured.fields[0].owner_path.is_canonical
    assert captured.diagnostics[0].block.owner_path.is_canonical
    assert diagnostics[0].block.owner_path.is_canonical
    assert _field_names(captured.fields) == ["transport"]

    # Snapshot capture never rewrites user authoring objects.
    assert policy.fields == [state_ref]
    assert policy.fields[0].owner_path.is_authoring
    assert policy.diagnostics[0] is measure
    assert measure.block is block


def test_runtime_snapshot_capture_refuses_ambiguous_model_local_reference():
    from pops.codegen.orchestration import _capture_runtime_declarations
    from pops.model import AmbiguousReferenceError

    module = Module("ambiguous-snapshot")
    state = module.state_space("U", components=("rho",))
    state_ref = module.state_handle(state)
    problem = pops.Problem(name="ambiguous-snapshot")
    problem.add_block("a", module)
    problem.add_block("b", module)
    problem.output(OutputPolicy(fields=[state_ref]))
    with pytest.raises(AmbiguousReferenceError, match="candidates"):
        _capture_runtime_declarations(problem)


def test_runtime_policies_exported_on_root():
    assert pops.RuntimePolicies is RuntimePolicies


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
