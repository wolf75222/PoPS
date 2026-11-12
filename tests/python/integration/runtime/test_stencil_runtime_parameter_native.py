"""A ``StencilFlux`` that reads a runtime parameter gives, from one artifact, the step of each value
bound to it (N3 lot 1).

The heat case of ``tests/python/support/stencil_runtime_parameter.py`` is compiled once and bound
with nu = 0.01 and nu = 0.05: each run takes one step u + dt nu sum_a (u_{+a} - 2 u + u_{-a}) / h_a^2
of its own value, so the two results differ, while the artifact is the same. Its unit counterpart
(``tests/python/unit/numerics/test_stencil_runtime_parameter.py``) shows that two defaults give the
same artifact hash and the same C++.
"""
import numpy as np
import pops
import pytest

from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile
from tests.python.support.stencil_runtime_parameter import heat_case

CELLS = 16
DT = 1.e-3


def _expected(initial, nu):
    h2 = (1. / CELLS) ** 2
    laplacian = sum(np.roll(initial, shift, axis=axis) for axis in (1, 2) for shift in (-1, 1)) \
        - 4. * initial
    return initial + DT * nu * laplacian / h2


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_one_artifact_steps_with_the_value_bound_to_the_parameter(isolated_native_cache,
                                                                  native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope="stencil runtime parameter")
    case, layout, block, state, nu = heat_case(cells=(CELLS, CELLS), dt=DT)
    x, y = np.meshgrid((np.arange(CELLS) + .5) / CELLS, (np.arange(CELLS) + .5) / CELLS)
    initial = (1. + .3 * np.cos(2 * np.pi * x) + .2 * np.sin(4 * np.pi * y))[None]
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_profile_plan_once(world, resolved, route="stencil runtime parameter",
                                         compile_artifact=pops.compile)
    results = {}
    for value in (0.01, 0.05):
        runtime = collective_call(world, lambda value=value: pops.bind(
            artifact, params={nu: value},
            initial_values={block[state]: np.ascontiguousarray(initial)},
            resources={"execution_context": artifact_execution_context(artifact)}))
        collective_call(world, lambda runtime=runtime: pops.run(runtime, t_end=DT, max_steps=1,
                                                                console=False))
        after = collective_call(world, lambda runtime=runtime: runtime.state_global("scalar"))
        if after is not None:
            results[value] = np.asarray(after).reshape(initial.shape)
    with collective_check(world):
        if results:
            for value, after in results.items():
                np.testing.assert_allclose(after, _expected(initial, value), rtol=0, atol=1e-13)
            assert np.max(np.abs(results[0.01] - results[0.05])) > 1e-6
