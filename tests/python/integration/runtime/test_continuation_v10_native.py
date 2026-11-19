"""Run only after integration/build. No result in this file is a native acceptance claim."""
import numpy as np
import pops
import pytest
from examples.api_v02.continuation_checks import build_continuation
from tests.python.integration.runtime.test_api_v02_methods_T01_T12 import _compile,_bind,_require_owned_array
from tests.python.support.collective_checks import collective_call,collective_check,collective_attempt
from tests.python.support.integral_state_receipts import collective_directory

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


@pytest.mark.parametrize('mode',('success','retry'))
def test_complete_continuation_recurs_without_mutating_retry_base(mode,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('Continuation_'+mode,builder=build_continuation,mode=mode,steps=3)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.3,max_steps=3,console=False))
    actual=collective_call(world,lambda:runtime.state_global('method'));_require_owned_array(world,actual)
    with collective_check(world):
        assert runtime.macro_step()==3 and abs(runtime.time()-.3)<1.e-14
        if actual is not None:np.testing.assert_allclose(actual,10.,atol=1.e-13,rtol=0)


@pytest.mark.parametrize('mode',('assessment_reject','stale_transform','wrong_candidate','parent_reject','duplicate_increment','loop_identity'))
def test_invalid_acceptance_never_publishes_any_outer_leaf(mode,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('Continuation_'+mode,builder=build_continuation,mode=mode,steps=1)
    runtime=_bind(spec,artifact,world)
    before=collective_call(world,lambda:runtime.state_global('method'));_require_owned_array(world,before)
    _,failures=collective_attempt(world,lambda:pops.run(runtime,t_end=.1,max_steps=1,console=False))
    after=collective_call(world,lambda:runtime.state_global('method'))
    with collective_check(world):
        assert failures and all(failures),failures
        assert runtime.time()==0. and runtime.macro_step()==0
        if before is not None:np.testing.assert_array_equal(before,after)


def test_full_array_variant_and_domain_aliases_survive_strict_restart(tmp_path,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile('Continuation_restart',builder=build_continuation,steps=4)
    uninterrupted=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(uninterrupted,t_end=.2,max_steps=2,console=False))
    before=collective_call(world,lambda:uninterrupted.state_global('method'));_require_owned_array(world,before)
    with collective_check(world):
        if before is not None:np.testing.assert_array_equal(before,4.)
    directory=collective_directory(world,tmp_path)
    checkpoint=collective_call(world,lambda:uninterrupted.checkpoint(str(directory/'Continuation')))
    restored=_bind(spec,artifact,world);collective_call(world,lambda:restored.restart(checkpoint))
    for runtime in (uninterrupted,restored):
        collective_call(world,lambda:pops.run(runtime,t_end=.4,max_steps=2,console=False))
    expected=collective_call(world,lambda:uninterrupted.state_global('method'))
    actual=collective_call(world,lambda:restored.state_global('method'))
    with collective_check(world):
        assert restored.macro_step()==4 and abs(restored.time()-.4)<1.e-14
        if actual is not None:
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(actual,20.)


@pytest.mark.parametrize(
    "dt,cut,split_steps,until,steps,segment_steps",
    (
        (0.125, 0.25, None, 0.5, 4, 2),
        (0.1, 0.27, None, 0.47, 5, 0),
        (0.05 / 16, 0.25, 17, 0.25, 80, 17),
    ),
)
def test_compiled_fixed_grid_survives_public_restart(
    dt, cut, split_steps, until, steps, segment_steps, tmp_path,
    isolated_native_cache, native_cxx, kokkos_root,
):
    from pops.runtime._simulation import BudgetExhausted, Simulation
    from pops.time import FixedDt
    from pops.time._program.control import AdvanceControl

    def scheduled():
        spec = build_continuation(steps=steps)
        spec.case._time.step_strategy(FixedDt(dt))
        spec.case._time._set_advance_control(AdvanceControl(
            assessments=(("assessment", "continuation"),),
            first_dt=dt, max_attempts=2, max_transforms=0,
        ))
        return spec

    spec, artifact, world = _compile("Continuation_compiled_grid", builder=scheduled)
    original = _bind(spec, artifact, world)
    simulation = Simulation(original)
    if split_steps is None:
        collective_call(world, lambda: simulation.run(until=cut))
    else:
        def split():
            with pytest.raises(BudgetExhausted) as exhausted:
                simulation.run(until=cut, max_steps=split_steps)
            assert exhausted.value.result.steps == split_steps

        collective_call(world, split)
    state = original._executor._temporal_restart_state
    grid = state.controller_state["fixed_dt_grid"]
    with collective_check(world):
        assert grid["steps"] == segment_steps
        assert grid["origin"] == (cut if segment_steps == 0 else 0.0).hex()
        assert grid["time"] == original.time().hex()
        assert grid["macro_step"] == original.macro_step()
        last_begin = 0.0
        for _ in range(original.macro_step() - 1):
            last_begin += dt
        assert state.controller_state["last_accepted_dt"] == (original.time() - last_begin).hex()
        manifest = original.last_run_manifest
        assert manifest.controls["step_transaction"]["compiled_advance"]["schema"] == (
            simulation.control.to_data()["schema"]
        )

    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: original.checkpoint(str(directory / "compiled")))
    restored = _bind(spec, artifact, world)
    collective_call(world, lambda: restored.restart(checkpoint))
    restored_state = restored._executor._temporal_restart_state
    actual = collective_call(world, restored._executor._s._checkpoint_continuation_image)
    expected = collective_call(world, original._executor._s._checkpoint_continuation_image)
    with collective_check(world):
        assert restored_state.controller_state == state.controller_state
        assert actual == expected

    resumed = Simulation(restored)
    empty = collective_call(world, lambda: resumed.run(until=restored.time()))
    with collective_check(world):
        assert empty.steps == 0
        assert restored_state.controller_state == state.controller_state
    for driver, runtime in ((simulation, original), (resumed, restored)):
        outcome = collective_call(world, lambda: driver.run(until=until))
        with collective_check(world):
            assert outcome.time == runtime.time() == until
            assert runtime.macro_step() == steps
    original_state = collective_call(world, lambda: original.state_global("method"))
    restored_values = collective_call(world, lambda: restored.state_global("method"))
    _require_owned_array(world, original_state)
    _require_owned_array(world, restored_values)
    with collective_check(world):
        if original_state is not None:
            np.testing.assert_array_equal(restored_values, original_state)
            np.testing.assert_array_equal(restored_values, steps * (steps + 1) * (steps + 2) // 6)
        assert restored_state.to_data() == state.to_data()
