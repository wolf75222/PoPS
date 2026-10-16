"""Shared native spatial receivers; execution receipts establish qualification."""
from __future__ import annotations
import json
import numpy as np
import pops
from examples.api_v02.scientific_spatial_v14 import build_scientific_spatial_v14
from tests.python.support.api_v02_native import _compile, _bind
from tests.python.support.api_v02_receipts import _decode, _image
from tests.python.support.api_v02_receipts import _shards
from tests.python.support.decision_assess_v14 import _error_value
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory

def _reference(count, steps, krylov="cg"):
    """Independent dense NumPy residual/Jacobian; no PoPS builder/helper calls."""
    q = 1+.1*np.arange(count, dtype=float); h=.02; mu=h*.1*count*count
    advection = (.3*h*count if krylov == "gmres" else 0)*(np.eye(count)-np.roll(np.eye(count), -1, axis=1))
    matrix = -2*np.eye(count)+np.roll(np.eye(count), 1, axis=1)+np.roll(np.eye(count), -1, axis=1)
    for _ in range(steps):
        old = q.copy()
        for _ in range(12):
            r = q-old-mu*(matrix@q)+advection@q+h*q**3
            if np.dot(r,r) < 1.e-29: break
            jacobian = np.eye(count)-mu*matrix+advection+np.diag(3*h*q*q)
            q -= np.linalg.solve(jacobian, r)
        assert np.dot(q-old-mu*(matrix@q)+advection@q+h*q**3, q-old-mu*(matrix@q)+advection@q+h*q**3) < 1.e-26
    return q

def _check(spec, image, step, count, krylov):
    root = _decode(spec, image)['root']
    assert root['step'] == step and abs(root['time']-.02*step) < 1.e-14
    assert root['q']['domain'] is root['domain'] and root['domain']['size'] == count
    actual = np.asarray(root['q']['values'])
    np.testing.assert_allclose(actual, _reference(count, step, krylov), rtol=0, atol=3.e-11)
    assert np.max(np.abs(actual-(1+.1*np.arange(count)))) > .01

def run_success(count, krylov, tmp_path, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile('N03_spatial_'+str(count)+'_'+krylov, builder=build_scientific_spatial_v14,
                                     count=count, steps=2, krylov=krylov)
    assert not spec.case._time._block_indices()
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda:pops.run(runtime, t_end=.02, max_steps=1, console=False))
    image = collective_call(world, lambda:_image(runtime))
    with collective_check(world): _check(spec, image, 1, count, krylov)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda:runtime.checkpoint(str(directory/'scientific_spatial')))
    restored = _bind(spec, artifact, world)
    collective_call(world, lambda:restored.restart(checkpoint))
    copied = collective_call(world, lambda:_image(restored))
    with collective_check(world): assert copied == image
    for target in (runtime, restored):
        collective_call(world, lambda:pops.run(target, t_end=.04, max_steps=1, console=False))
        image = collective_call(world, lambda:_image(target))
        with collective_check(world): _check(spec, image, 2, count, krylov)

def run_failure(isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile('N03_false_inner', builder=build_scientific_spatial_v14,
                                     count=5, steps=1, dishonest=True)
    runtime = _bind(spec, artifact, world)
    error = None
    try:
        pops.run(runtime, t_end=.02, max_steps=1, console=False)
    except Exception as caught:
        error = caught
    with collective_check(world):
        from pops._bootstrap import StepAttemptRejected
        assert isinstance(error, StepAttemptRejected), repr(error)
        assert runtime.macro_step() == 0 and runtime.time() == 0
        signature = json.loads(error.typed_failure_schema)['type']
        shards = _shards(error.typed_failure_payload)
        assert len(shards) == (1 if world is None else int(world.size))
        for shard in shards:
            # ORIGINAL_RESIDUAL: the independently re-evaluated inner equation
            # rejects the zero correction returned as success by the algorithm.
            assert _error_value(shard, signature) == {'tag':'relation', 'value':{'code':2107}}
