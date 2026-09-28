"""Public SymbolicPath execution on explicit synchronous AMR hierarchies.

The one-level case has an independent full-array FV oracle. The two-level case
requires actual partial refinement, verifies its fine-cell oracle, and checks
the native composite integral of the conservative component.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest

from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import CoarseFineInjection, ConservativeInjection, StateTransfer
from pops.lib.initial import BindArray
from pops.lib.time import ForwardEuler
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import (DiscretizationPlan, PathConservativeFiniteVolume,
                           reconstruction, riemann, variables)
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import AdaptiveCFL, StagePoint, TimePoint
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.symbolic_path_case import declarations


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
N = 16
DT = 1.e-4


def _resolved(levels, *, hot_stage=False):
    model, state, flux, product, path = declarations()
    rate = model.rate("balance", equation=ddt(state) == -div(flux) - product)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, PathConservativeFiniteVolume(
        flux=flux, path=path, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case = pops.Case("symbolic-path-amr-%d-%s" % (levels, "hot" if hot_stage else "normal"))
    block = case.block("transport", model)
    instance = block[state]
    case.numerics(numerics, block=block)
    if hot_stage:
        program = pops.Program("hot-path-stage")
        temporal = program.state(instance)
        stage = StagePoint("hot", {"main": TimePoint(program.clock, 0)})
        hot = program.value("hot_input", 100. * temporal.n, at=stage)
        rhs = program.value("path_rhs", rate(hot), at=stage)
        result = program.value("next", temporal.n + program.dt * rhs, at=temporal.next.point)
        program.commit(temporal.next, result)
    else:
        program = ForwardEuler(instance, rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=.25, max_dt=DT))
    case.program(program)
    case.initials.add(InitialCondition(state=instance, value=BindArray(),
                                      projection=ConservativeCellAverage()))
    transfer = AMRTransfer()
    transfer.state(instance, StateTransfer(prolongation=ConservativeInjection(),
                                          coarse_fine=CoarseFineInjection()))
    threshold = case.param(RuntimeParam("refine_u", default=1.05))
    layout = AMR(
        grid=CartesianGrid(frame=path.frame, cells=(N, N),
                           periodic=PeriodicAxes(path.frame.axes)),
        hierarchy=AMRHierarchy(max_levels=levels, ratios=(2,) * (levels - 1)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(instance)["u"] > case.value(threshold)), Buffer(cells=0)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid.frozen(), transfer=transfer, execution=AMRExecution.synchronous())
    return pops.resolve(pops.validate(case), layout=layout), instance


def _initial():
    x = (np.arange(N) + .5) / N
    u = np.broadcast_to(1. + .1 * np.sin(2. * np.pi * x), (N, N))
    v = np.broadcast_to(.3 + .1 * np.cos(2. * np.pi * x), (N, N))
    return np.ascontiguousarray(np.stack((u, v)))


def _one_step(initial, cells):
    """Analytic integral of v(s)^2 du(s), independent of symbolic quadrature."""
    right = np.roll(initial, -1, axis=2)
    speed = .5 + np.maximum(initial[1] ** 2, right[1] ** 2)
    flux = .25 * (initial + right) - .5 * speed * (right - initial)
    action = np.zeros_like(initial)
    action[0] = (initial[1] ** 2 + initial[1] * right[1] + right[1] ** 2) \
        * (right[0] - initial[0]) / 3.
    rhs = cells * (np.roll(flux, 1, axis=2) - flux
                   - .5 * (action + np.roll(action, 1, axis=2)))
    return initial + DT * rhs


def _bind(levels, *, hot_stage=False):
    resolved, state = _resolved(levels, hot_stage=hot_stage)
    assert resolved.target == "amr_system"
    artifact = pops.compile(resolved)
    initial = _initial()
    runtime = pops.bind(artifact, initial_values={state: initial},
                        resources={"execution_context": artifact_execution_context(artifact)})
    assert runtime.n_levels() == levels
    coarse = np.asarray(runtime.block_level_state_global("transport", 0)).reshape(initial.shape)
    np.testing.assert_array_equal(coarse, initial)
    return runtime, initial


def test_symbolic_path_one_level_amr_matches_independent_oracle(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    runtime, initial = _bind(1)
    assert runtime.patch_boxes() == []
    mass_v = runtime.integral("transport", component=1)
    mass_u = runtime.integral("transport", component=0)
    report = pops.run(runtime, t_end=DT, max_steps=1)
    assert report.accepted_steps == runtime.macro_step() == 1
    assert runtime.time() == pytest.approx(DT, rel=0., abs=1.e-15)
    actual = np.asarray(runtime.block_level_state_global("transport", 0)).reshape(initial.shape)
    np.testing.assert_allclose(actual, _one_step(initial, N), rtol=0., atol=5.e-14)
    assert runtime.integral("transport", component=1) == pytest.approx(mass_v, abs=5.e-14, rel=0.)
    # Conservative advection alone cannot change the periodic integral of u.
    assert abs(runtime.integral("transport", component=0) - mass_u) > 1.e-7


def test_symbolic_path_partial_two_level_amr_conserves_v_and_evolves_nonconservative_u(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    runtime, initial = _bind(2)
    boxes = runtime.patch_boxes()
    fine_mask = np.zeros((2 * N, 2 * N), dtype=bool)
    for level, lower, upper in boxes:
        assert level == 1
        fine_mask[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1] = True
    assert 0 < np.count_nonzero(fine_mask) < fine_mask.size, "partial fine hierarchy required"
    injected = np.repeat(np.repeat(initial, 2, axis=1), 2, axis=2)
    fine_before = np.asarray(runtime.block_level_state_global("transport", 1)).reshape(injected.shape)
    np.testing.assert_allclose(fine_before[:, fine_mask], injected[:, fine_mask], rtol=0., atol=1.e-15)
    mass_v = runtime.integral("transport", component=1)
    mass_u = runtime.integral("transport", component=0)
    assert mass_v == pytest.approx(initial[1].mean(), rel=0., abs=5.e-14)
    report = pops.run(runtime, t_end=DT, max_steps=1)
    assert report.accepted_steps == runtime.macro_step() == 1
    assert runtime.time() == pytest.approx(DT, rel=0., abs=1.e-15)
    assert runtime.patch_boxes() == boxes
    fine = np.asarray(runtime.block_level_state_global("transport", 1)).reshape(injected.shape)
    np.testing.assert_allclose(fine[:, fine_mask], _one_step(injected, 2 * N)[:, fine_mask],
                               rtol=0., atol=8.e-14)
    assert runtime.integral("transport", component=1) == pytest.approx(mass_v, rel=0., abs=8.e-14)
    assert abs(runtime.integral("transport", component=0) - mass_u) > 1.e-7


def test_symbolic_path_actual_stage_cfl_failure_keeps_two_level_accepted_state(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    runtime, _ = _bind(2, hot_stage=True)
    before = tuple(np.asarray(runtime.block_level_state_global("transport", level)).copy()
                   for level in range(2))
    boxes = runtime.patch_boxes()
    assert boxes
    with pytest.raises(RuntimeError, match="actual face CFL exceeds authored Courant"):
        pops.run(runtime, t_end=DT, max_steps=1)
    assert runtime.time() == 0.
    assert runtime.macro_step() == 0
    assert runtime.patch_boxes() == boxes
    for level, accepted in enumerate(before):
        np.testing.assert_array_equal(
            np.asarray(runtime.block_level_state_global("transport", level)), accepted)
