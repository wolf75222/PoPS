"""Collective native reception helper; this module defines no collected tests."""
from dataclasses import dataclass
import hashlib
import json
import pops

from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile, profile_gather_value


def _compile(method, *, builder=None, **kwargs):
    from examples.api_v02.methods_T01_T12 import build_method
    _native, world = native_profile(dimension=2, scope="API v0.2 "+method+" method")
    spec = collective_call(world, lambda: builder(**kwargs) if builder is not None
                           else build_method(method, **kwargs))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, resolved, route="API v0.2 "+method,
                                         compile_artifact=pops.compile)
    return spec, artifact, world



def _bind(spec, artifact, world):
    return collective_call(world, lambda: pops.bind(artifact, initial_values={
        subject: value.copy() for subject,value in spec.initial_values.items()},
        resources={"execution_context":artifact_execution_context(artifact)}))



def execute_case(builder,world,route,*,return_report=False,max_steps=32):
    spec=collective_call(world,builder)
    plan=collective_call(world,lambda:pops.resolve(pops.validate(spec.case),layout=spec.layout))
    artifact=compile_profile_plan_once(world,plan,route=route,compile_artifact=pops.compile)
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values=dict(spec.initial_values),
        resources={"execution_context":artifact_execution_context(artifact)}))
    report=collective_call(world,lambda:pops.run(runtime,t_end=spec.t_end,max_steps=max_steps,console=False))
    metrics=collective_call(world,lambda:spec.invariants(runtime))
    gathered=profile_gather_value(world,metrics)
    with collective_check(world):
        assert all(row["passed"] for row in gathered),gathered
        assert abs(runtime.time()-spec.t_end)<2e-13
    return (runtime,artifact,gathered,report) if return_report else (runtime,artifact,gathered)


@dataclass(frozen=True)
class ContinuationSnapshot:
    """A configured native frontier is explicitly initialized or uninitialized."""
    status: str
    schema: str
    capacity: int
    time: float
    macro_step: int
    image: bytes | None
    checkpoint_error: str | None


def continuation_snapshot(runtime, world):
    """Call collectively; only the exact native uninitialized refusal is admitted.

    This does not initialize the store or weaken the existing strict _image helper.
    An empty byte string is never used as an uninitialized-image surrogate.
    """
    native = runtime._executor._s
    ranks = 1 if world is None else int(world.size)
    expected = ("continuation has not been initialized" if ranks == 1 else
                "Continuation checkpoint capture; rank 0: continuation has not been initialized")
    image = None
    diagnostic = None
    try:
        image = native._checkpoint_continuation_image()
    except RuntimeError as error:
        if type(error) is not RuntimeError or str(error) != expected:
            raise
        diagnostic = str(error)
    # These are local observations, after the checkpoint's collective boundary.
    schema = native._checkpoint_continuation_schema()
    capacity = native._checkpoint_continuation_capacity()
    assert type(schema) is str and schema
    assert type(capacity) is int and capacity > 0
    if diagnostic is None:
        assert type(image) is bytes and image, "initialized Continuation requires a complete image"
    else:
        assert image is None
    return ContinuationSnapshot("uninitialized" if diagnostic is not None else "initialized",
        schema, capacity, runtime.time(), runtime.macro_step(), image, diagnostic)


def recorded_attempt(world, operation):
    """Retain the actual local exception in addition to converged diagnostics."""
    errors = []
    def attempt():
        try:
            return operation()
        except Exception as error:
            errors.append(error)
            raise
    _, failures = collective_attempt(world, attempt)
    return errors[0] if errors else None, failures


def typed_failure_values(error, world):
    """Check exception/envelope identities and decode every exact scalar failure shard."""
    from pops._bootstrap import StepAttemptRejected
    from tests.python.support.api_v02_receipts import _shards, _error_value
    assert isinstance(error, StepAttemptRejected), repr(error)
    schema = json.loads(error.typed_failure_schema)
    assert schema['schema'] == 'pops.typed-failure@1'
    signature = schema['type']
    shards = _shards(error.typed_failure_payload)
    assert len(shards) == (1 if world is None else int(world.size))
    return signature, tuple(_error_value(shard, signature) for shard in shards)


def assert_typed_failure(error, world, *, tag, code):
    _, payloads = typed_failure_values(error, world)
    assert all(value == {'tag': tag, 'value': {'code': code}} for value in payloads), payloads


def assert_requests_surface_failure(error, world):
    signature, payloads = typed_failure_values(error, world)
    # The transformed Requests body rejects in its actual local map, whose
    # failure contains the first failing source rank/gid and the complete cause.
    # Both direct Requests and Requests inside iterate preserve this transport
    # envelope; a bare differential tag would discard the origin and is rejected.
    for payload in payloads:
        tag = payload['tag']
        child_signature = dict(signature['members'])[tag]
        digest = hashlib.sha256(json.dumps(child_signature, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        assert tag == 'differential_transport_' + digest
        assert payload['value'] == {'rank': 0, 'gid': 0, 'reason': {
            'tag': 'user', 'value': {'tag': 'differential', 'value': {'code': 91001}}}}, payload


def assert_legacy_failure(error, world, *, origin, code):
    """Exact public-step Result rejection, including one ordered cause per rank."""
    from pops._bootstrap import StepAttemptRejected
    assert isinstance(error, StepAttemptRejected), repr(error)
    assert not error.typed_failure_schema and not error.typed_failure_payload
    ranks = 1 if world is None else int(world.size)
    assert ranks > 0
    detail = f"Result failed; successful value unavailable; origin={origin}; code={code}"
    assert error.status == "invalid_evaluation"
    assert error.phase == "structured_control" and error.disposition == "reject"
    assert error.reason_code == 0 and error.detail == detail
    local = ("step attempt rejected during structured_control: solve status=invalid_evaluation "
             f"(attempt_action=reject, reason_code=0, {detail})")
    # The native step boundary preserves all fields through its typed collective
    # rejection envelope. Python aggregates one complete cause per rank for MPI.
    expected = (local if ranks == 1 else
                "collective step attempt rejected during structured_control: " +
                "; ".join(f"rank {rank} StepAttemptRejected: {local}" for rank in range(ranks)))
    assert str(error) == expected, str(error)
