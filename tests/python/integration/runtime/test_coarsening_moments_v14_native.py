"""Independent N01 moment obstruction and variance-enriched native receiver.

Collection alone is not native validation. Run against the matching installed
Program engine in serial/MPI2/MPI3, with all ranks participating in each check.
"""
import json
import numpy as np
import pops
import pytest

from examples.api_v02.coarsening_moments_v14 import build_coarsening_moments_v14
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import _compile, _bind
from tests.python.integration.runtime.test_bundle_remap_v14_native import _decode, _image
from tests.python.integration.runtime.test_api_v02_terminal_failure_native import _shards
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.decision_assess_v14 import _error_value
from tests.python.support.integral_state_receipts import collective_directory

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _check(spec, image, step, *, enriched):
    root = _decode(spec, image)['root']; bundle = root['bundle']
    assert root['step'] == step and abs(root['time']-.1*step) < 1.e-14
    coarse = step >= 2
    current, source = bundle['instance'], bundle['maps']['source']
    assert current is not source and current['size'] == (1 if coarse else 2)
    assert source['size'] == (2 if step <= 2 else 1)
    q = np.asarray(bundle['fields']['q']['values'])
    np.testing.assert_array_equal(q, [1.] if coarse else [0., 2.])
    volume = np.asarray(bundle['geometry']['volume']['values'])
    np.testing.assert_array_equal(volume, [1.] if coarse else [.5, .5])
    np.testing.assert_array_equal(bundle['geometry']['center']['values'], [.5] if coarse else [.25, .75])
    variance = np.asarray(bundle['fields']['variance']['values']) if enriched else np.zeros_like(q)
    if enriched:
        np.testing.assert_array_equal(variance, [1.] if coarse else [0., 0.])
    else:
        assert set(bundle['fields']) == {'q'}
    assert np.dot(volume, q) == 1 and np.dot(volume, q*q+variance) == 2
    assert np.dot(volume, q+q*q+variance) == 3
    if coarse:
        assert enriched and q[0]*q[0] == 1 != 2 and variance[0] == 1
    # The retained sample (1,3) is independent of the current state (0,2).
    expected_history = {'mean': [2.] if coarse else [1., 3.],
        'second': [5.] if coarse else [1., 9.],
        'accumulation': [7.] if coarse else [2., 12.]}
    for key, expected in expected_history.items():
        np.testing.assert_array_equal(bundle['histories'][key]['values'], expected)
    for array in (*bundle['fields'].values(), *bundle['geometry'].values(),
                  *bundle['histories'].values(), *[v for k, v in bundle['maps'].items() if k != 'source']):
        assert array['domain'] is current
    for j, (left, right) in enumerate(zip(bundle['maps']['left']['values'], bundle['maps']['right']['values'])):
        assert left['domain'] is source and right['domain'] is source
        assert left['offset'] == (0 if step == 2 else j)
        assert right['offset'] == (1 if step == 2 else j)
    np.testing.assert_array_equal(bundle['maps']['left_weight']['values'], [.5] if step == 2 else np.ones(len(q)))
    np.testing.assert_array_equal(bundle['maps']['right_weight']['values'], [.5] if step == 2 else np.zeros(len(q)))
    return bundle


def test_mean_only_coarsening_rejects_original_second_moment_and_preserves_record(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile('N01_mean_only', builder=build_coarsening_moments_v14,
                                      mode='mean_only', steps=2)
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(runtime, t_end=.1, max_steps=1, console=False))
    before = collective_call(world, lambda: _image(runtime))
    with collective_check(world): _check(spec, before, 1, enriched=False)
    error = None
    try:
        pops.run(runtime, t_end=.2, max_steps=1, console=False)
    except Exception as caught:
        error = caught
    after = collective_call(world, lambda: _image(runtime))
    with collective_check(world):
        from pops._bootstrap import StepAttemptRejected
        assert isinstance(error, StepAttemptRejected), repr(error)
        assert runtime.time() == .1 and runtime.macro_step() == 1 and after == before
        _check(spec, after, 1, enriched=False)
        signature = json.loads(error.typed_failure_schema)['type']
        shards = _shards(error.typed_failure_payload)
        assert len(shards) == (1 if world is None else int(world.size))
        for shard in shards:
            payload = _error_value(shard, signature)
            assert payload == {'tag': 'moment_loss', 'value': {
                'mean': 0., 'second': -1., 'accumulation': -1., 'volume': 0.,
                'history_mean': 0., 'history_second': 0., 'history_accumulation': 0.}}


def test_variance_enrichment_preserves_both_moments_complete_history_and_restart(
        tmp_path, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile('N01_enriched', builder=build_coarsening_moments_v14,
                                      mode='enriched', steps=3)
    assert not spec.case._time._block_indices() and not spec.initial_values
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(runtime, t_end=.1, max_steps=1, console=False))
    fine = collective_call(world, lambda: _image(runtime))
    with collective_check(world): _check(spec, fine, 1, enriched=True)
    collective_call(world, lambda: pops.run(runtime, t_end=.2, max_steps=1, console=False))
    coarse = collective_call(world, lambda: _image(runtime))
    with collective_check(world):
        _check(spec, coarse, 2, enriched=True)
        assert fine != coarse
    directory = collective_directory(world, tmp_path)
    path = collective_call(world, lambda: runtime.checkpoint(str(directory/'coarsening_moments')))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(path))
    restored = collective_call(world, lambda: _image(resumed))
    with collective_check(world):
        assert restored == coarse
        _check(spec, restored, 2, enriched=True)
    final = []
    for system in (runtime, resumed):
        collective_call(world, lambda: pops.run(system, t_end=.3, max_steps=1, console=False))
        final.append(collective_call(world, lambda: _image(system)))
    with collective_check(world):
        a, b = [_check(spec, image, 3, enriched=True) for image in final]
        for role in ('fields', 'geometry', 'histories'):
            for name in a[role]:
                np.testing.assert_array_equal(a[role][name]['values'], b[role][name]['values'])
