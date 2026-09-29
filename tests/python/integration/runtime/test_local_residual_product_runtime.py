"""Installed native acceptance of a public heterogeneous local unknown product."""
import numpy as np
import pops
import pytest
from tests.python.support.local_residual_product_case import make_case
from tests.python.support.native_execution_context import artifact_execution_context
from test_user_numerical_bodies_runtime import _compile, _root_check

from tests.python.support.collective_checks import (
    collective_attempt, collective_call, collective_check, state_snapshots,
)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def snapshots(runtime, world):
    return state_snapshots(runtime, world, ("left", "right"))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_joint_product_rebind_or_atomic_failure(
        isolated_native_cache, native_cxx, kokkos_root, reverse, invalid):
    case, layout, subjects = make_case(reverse=reverse, invalid=invalid)
    artifact, world = _compile(case, layout, "local-product-%s-%s" % (reverse, invalid))
    for capture in ((2.,3.,7.,11.,5.),(-1.,4.,2.,9.,3.)):
        with collective_check(world):
            a = np.broadcast_to(np.array(capture[:2])[:,None,None], (2,4,4)).copy()
            b = np.broadcast_to(np.array(capture[2:])[:,None,None], (3,4,4)).copy()
        runtime = collective_call(world, lambda: pops.bind(artifact, initial_values=dict(zip(subjects,(a,b),strict=True)),
                            resources={"execution_context": artifact_execution_context(artifact)}))
        before = snapshots(runtime, world)
        _, failures = collective_attempt(world, lambda: pops.run(
            runtime, t_end=.01, max_steps=1, console=False))
        with collective_check(world):
            if invalid:
                assert all(row is not None and row[2] for row in failures), failures
            else:
                assert not any(failures), failures
        after = snapshots(runtime, world)
        if invalid:
            with collective_check(world):
                assert runtime.time() == 0. and runtime.macro_step() == 0
            def unchanged(before=before, after=after):
                for old,new in zip(before,after,strict=True):
                    np.testing.assert_array_equal(new,old)
            _root_check(world, unchanged)
        else:
            with collective_check(world):
                assert runtime.macro_step() == 1
                assert runtime.time() == pytest.approx(.01)
            def original_equations(after=after, a=a, b=b):
                x=after[0].reshape(2,4,4)
                z=after[1].reshape(3,4,4)
                residual=np.concatenate((z[:2]-a,(x[0]*z[0]-b[0])[None,:,:],
                                         (x[1]+z[1]-b[1])[None,:,:],z[2:]-b[2:]))
                assert np.max(np.abs(residual)) < 1e-11
                np.testing.assert_allclose(x,np.stack((b[0]/a[0],b[1]-a[1])),rtol=0,atol=1e-10)
            _root_check(world, original_equations)
