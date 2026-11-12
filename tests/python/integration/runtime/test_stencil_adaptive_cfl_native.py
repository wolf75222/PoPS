"""b6's one stability guard (503) covers a ``StencilFlux`` beside a Python face (N3 lot 1, O4).

The case is L1 of §17.1 in two dimensions, as N3 lot 1 elaborates it: transport by the b6 Rusanov
face (a ``riemann.User`` face, the face of ``upwind()``) and the two-point diffusion of
``two_point()`` as a ``StencilFlux`` whose incident frequency is (nu / h) / h per face (D10). Its
cell sum is 4 nu / h^2 on the uniform square, and the face frequency of the transport is 2 a / h
under ``FixedDt`` (the incident sum).

* Under ``FixedDt``, the guard reads the frequency of the stencil: with a = 1 and h = 1/16, the
  step dt = 0.02 is admitted for nu = 0.001 (dt (2 a / h + 4 nu / h^2) = 0.66) and refused with
  the reason 503 for nu = 0.05 (1.66), which dt = 0.01 admits (0.83). The value of nu is bound to
  one artifact for each step (CF15).
* Under ``AdaptiveCFL``, the first proposal is master's cfl h / max s, which does not count the
  frequency of the stencil, while the guard does: it is refused (503). Since ROM-108 (#769) the
  refusal carries the frequency that the faces and the stencil recorded at the beginning state,
  a / h + 4 nu / h^2, and AdaptiveCFL retries once at cfl over it, which the guard admits: one
  step accepted at its second attempt (O4 of the review of the lot note; before ROM-108 the run
  stopped at its first step).
"""
import numpy as np
import pops
import pytest

from tests.python.support.collective_checks import collective_call
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile
from tests.python.support.stencil_runtime_parameter import transport_heat_case

CELLS = 16


def _initial():
    x, y = np.meshgrid((np.arange(CELLS) + .5) / CELLS, (np.arange(CELLS) + .5) / CELLS)
    return np.ascontiguousarray((1. + .3 * np.cos(2 * np.pi * x)
                                 + .2 * np.sin(4 * np.pi * y))[None])


def _artifact(world, strategy, name):
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage

    case, layout, block, state, nu = transport_heat_case(name, cells=(CELLS, CELLS),
                                                         strategy=strategy)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_profile_plan_once(world, resolved, route="stencil adaptive cfl",
                                         compile_artifact=pops.compile)
    return artifact, block[state], nu


def _bound(world, artifact, subject, nu, value):
    return collective_call(world, lambda: pops.bind(
        artifact, params={nu: value}, initial_values={subject: _initial()},
        resources={"execution_context": artifact_execution_context(artifact)}))


def _step(world, artifact, subject, nu, value, t_end):
    runtime = _bound(world, artifact, subject, nu, value)
    collective_call(world, lambda: pops.run(runtime, t_end=t_end, max_steps=1, console=False))


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_the_guard_counts_the_frequency_of_the_stencil_under_a_fixed_step(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops.time import FixedDt

    _native, world = native_profile(dimension=2, scope="stencil adaptive cfl")
    coarse = _artifact(world, FixedDt(0.02), "stencil_guard_coarse")
    _step(world, *coarse, 0.001, 0.02)
    with pytest.raises(Exception, match="reason_code=503"):
        _step(world, *coarse, 0.05, 0.02)
    _step(world, *_artifact(world, FixedDt(0.01), "stencil_guard_fine"), 0.05, 0.01)


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_adaptive_cfl_takes_the_step_that_the_guard_admits_beside_a_stencil(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops.time import AdaptiveCFL

    nu = 0.001
    _native, world = native_profile(dimension=2, scope="stencil adaptive cfl")
    artifact, subject, handle = _artifact(world, AdaptiveCFL(cfl=0.4), "stencil_guard_adaptive")
    runtime = _bound(world, artifact, subject, handle, nu)
    report = collective_call(world, lambda: pops.step(runtime, t_bound=1.0, console=False))
    clock = collective_call(world, lambda: float(runtime.time()))
    # the step cfl / (a / h + 4 nu / h^2) of the recorded frequency, at the second attempt
    assert (report.accepted_steps, report.rejected_steps, clock) == (
        1, 1, pytest.approx(0.4 / (CELLS + 4 * nu * CELLS ** 2), rel=1e-12))
