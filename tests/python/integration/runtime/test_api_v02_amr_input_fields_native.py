"""Declared input fields are read on their actual AMR level and survive restart."""
import numpy as np
import pops
import pytest
from pops import math
from pops.amr import (
    AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer, Buffer,
    ConflictPolicy, EqualityPolicy, Hysteresis, Tag,
)
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Analytic
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model.provider_pack import ProviderPack
from pops.numerics import Diffusion, DiscretizationPlan
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every
from tests.python.support.collective_checks import (
    collective_attempt, collective_call, collective_check,
)
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
N = 8
DT = 1.e-5


def _build():
    frame = Rectangle("input-field-square", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("input-field-heat", frame=frame)
    state = model.state("U", components=("u",))
    coefficient = model.aux("a")
    flux = model.diffusive_flux("flux", state=state, value=coefficient * math.grad(state))
    rate = model.rate("rate", equation=math.ddt(state) == math.div(flux))
    case = pops.Case("input-field-hierarchy")
    block = case.block("heat", model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, Diffusion(flux=flux))
    case.numerics(numerics, block=block)
    p = pops.Program("input-field-step")
    q = p.state(block[state])
    rhs = rate(q.n, p.input_fields(q.n, for_rate=rate))
    p.commit(q.next, p.value("next", q.n+p.dt*rhs, at=q.next.point))
    p.step_strategy(FixedDt(DT))
    case.program(p)
    from pops.analytic import cos, sin, x, y
    initial = Analytic(frame=frame, components=(
        2.+.2*cos(2*np.pi*x(frame))*sin(2*np.pi*y(frame)),))
    case.initials.add(InitialCondition(state=block[state], value=initial,
                                      projection=ConservativeCellAverage()))
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("refine_threshold", default=-100.))
    layout = AMR(
        grid=CartesianGrid(frame=frame, cells=(N, N), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(math.ValueExpr(block[state])[0] > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1000, clock=p.clock)),
        transfer=transfer, execution=AMRExecution.synchronous())
    return pops.resolve(pops.validate(case), layout=layout)


def _setup(world):
    plan = collective_call(world, _build)
    artifact = compile_profile_plan_once(world, plan, route="actual AMR runtime input fields",
                                         compile_artifact=pops.compile)
    pack = ProviderPack.from_data(plan.blocks[0].resolved_operations.to_data()
                                  ["provider_evidence"]["auxiliary"])
    keys = [key for key in pack if pack.declared_entry(key).producer == "runtime_input"]
    assert len(keys) == 1 and keys[0].component == "a"
    def bind():
        return pops.bind(artifact, aux={keys[0]: np.full((N, N), 2.)},
            resources={"execution_context": artifact_execution_context(artifact)})

    return artifact, keys[0], bind, collective_call(world, bind)


def _fine(runtime, world):
    values = collective_call(world, lambda: runtime.block_level_state_global("heat", 1))
    with collective_check(world):
        assert runtime.n_levels() == 2
    return None if values is None else np.asarray(values).reshape(2*N, 2*N).copy()


def _expected(values, coefficient):
    laplacian = sum(np.roll(values, shift, axis=axis)-values
                    for axis in (0, 1) for shift in (-1, 1))*(2*N)**2
    return values+DT*coefficient*laplacian


def test_refined_input_values_refresh_and_restart_with_the_discrete_equation(
        tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="API0.2 nonempty AMR input_fields")
    artifact, key, bind, runtime = _setup(world)
    initial = _fine(runtime, world)
    collective_call(world, lambda: pops.step(runtime, t_bound=DT, console=False))
    first = _fine(runtime, world)
    with collective_check(world):
        if initial is not None:
            np.testing.assert_allclose(first, _expected(initial, 2.), rtol=0, atol=3.e-13)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"input_fields")))
    resumed = collective_call(world, bind)
    collective_call(world, lambda: resumed.restart(checkpoint))
    restored = _fine(resumed, world)
    with collective_check(world):
        if first is not None:
            np.testing.assert_array_equal(restored, first)
    results = []
    for candidate in (runtime, resumed):
        collective_call(world, lambda candidate=candidate:
            candidate._executor.stage_auxiliary_input(key, np.full((N, N), 3.)))
        collective_call(world, lambda candidate=candidate:
            pops.step(candidate, t_bound=2*DT, console=False))
        results.append(_fine(candidate, world))
    with collective_check(world):
        if first is not None:
            np.testing.assert_allclose(results[0], _expected(first, 3.), rtol=0, atol=3.e-13)
            np.testing.assert_array_equal(results[0], results[1])
        assert runtime.time() == resumed.time() == 2*DT
    record_property("artifact", artifact.artifact_identity.hexdigest)


def test_rejected_refined_input_preserves_all_state_and_can_retry(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="API0.2 AMR input rejection")
    _, key, _, runtime = _setup(world)
    before = [collective_call(world, lambda level=level:
        runtime.block_level_state_global("heat", level)) for level in (0, 1)]
    collective_call(world, lambda:
        runtime._executor.stage_auxiliary_input(key, np.full((N, N), -1.)))
    _, errors = collective_attempt(world, lambda: pops.step(runtime, t_bound=DT, console=False))
    after = [collective_call(world, lambda level=level:
        runtime.block_level_state_global("heat", level)) for level in (0, 1)]
    with collective_check(world):
        assert errors and all(errors)
        assert runtime.time() == 0. and runtime.macro_step() == 0
        for old, new in zip(before, after, strict=True):
            if old is not None:
                np.testing.assert_array_equal(old, new)
    collective_call(world, lambda:
        runtime._executor.stage_auxiliary_input(key, np.full((N, N), 2.)))
    collective_call(world, lambda: pops.step(runtime, t_bound=DT, console=False))
    with collective_check(world):
        assert runtime.time() == DT and runtime.macro_step() == 1
