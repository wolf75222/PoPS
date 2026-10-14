"""Actual PoPS compile/bind/run reception of explicitly bounded method profiles."""
from __future__ import annotations

import math
import numpy as np
import pops
import pytest
from tests.python.support.api_v02_native import _compile, _bind

from examples.api_v02.common import MissingCapabilityError
from examples.api_v02.methods_T01_T12 import READY, build_method
from examples.api_v02.coverage import OBLIGATIONS
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile, profile_gather_value
from tests.python.support.integral_state_receipts import collective_directory




def _assert_metrics(world, rows):
    with collective_check(world):
        owners = [row for row in rows if row.get("metrics_owner", True)]
        assert owners and all(row["passed"] for row in owners), rows


def _require_owned_array(world, value):
    owners = profile_gather_value(world, value is not None)
    with collective_check(world):
        assert any(owners), "No rank returned the requested native state/history"
        if value is not None:
            assert np.asarray(value).size > 0
            assert np.isfinite(value).all()


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("method", READY)
def test_api_v02_native_method_profile(method, isolated_native_cache, native_cxx, kokkos_root,
                                        record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile(method)
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end, max_steps=32, console=False))
    metrics = collective_call(world, lambda: spec.invariants(runtime))
    rows = profile_gather_value(world,metrics)
    _assert_metrics(world, rows)
    with collective_check(world):
        assert abs(runtime.time()-spec.t_end) < 2.e-13
        assert runtime.macro_step() >= 1
        assert artifact.resolved_dimension == 2
    record_property("method_profile",spec.profile)
    record_property("native_metrics",str(rows))


@pytest.mark.compiler
@pytest.mark.native_loader
def test_t07_real_history_survives_strict_restart(tmp_path, isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile("T07")
    uninterrupted=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(uninterrupted,t_end=.1,max_steps=1,console=False))
    saved_state=collective_call(world,lambda:uninterrupted.state_global("method"))
    _require_owned_array(world,saved_state)
    saved_histories={}
    for name in ("bdf_state","bdf_increment"):
        saved_histories[name]=collective_call(world,lambda name=name:uninterrupted.history_global(name,1))
        _require_owned_array(world,saved_histories[name])
    with collective_check(world):
        assert uninterrupted.time()==.1 and uninterrupted.macro_step()==1
        if saved_state is not None:
            expected=np.broadcast_to(np.array([1/1.1,.1])[:,None,None],(2,4,4))
            np.testing.assert_allclose(np.asarray(saved_state).reshape(2,4,4),expected,atol=2.e-14,rtol=0)
            for name,expected_u in (("bdf_state",1.),("bdf_increment",1/1.1-1)):
                expected=np.broadcast_to(np.array([expected_u,0.])[:,None,None],(2,4,4))
                np.testing.assert_allclose(np.asarray(saved_histories[name]).reshape(2,4,4),expected,atol=2.e-14,rtol=0)
    directory=collective_directory(world,tmp_path)
    checkpoint=collective_call(world,lambda:uninterrupted.checkpoint(str(directory/"BDF2")))
    restarted=_bind(spec,artifact,world)
    collective_call(world,lambda:restarted.restart(checkpoint))
    restored_state=collective_call(world,lambda:restarted.state_global("method"))
    _require_owned_array(world,restored_state)
    with collective_check(world):
        assert restarted.time()==.1 and restarted.macro_step()==1
        if saved_state is not None:np.testing.assert_array_equal(saved_state,restored_state)
    for name,saved in saved_histories.items():
        restored=collective_call(world,lambda name=name:restarted.history_global(name,1))
        _require_owned_array(world,restored)
        with collective_check(world):
            if saved is not None:np.testing.assert_array_equal(saved,restored)
    for runtime in (uninterrupted,restarted):
        collective_call(world,lambda:pops.run(runtime,t_end=.25,max_steps=2,console=False))
        _assert_metrics(world,profile_gather_value(world,collective_call(world,lambda:spec.invariants(runtime))))
        diagnostics=collective_call(world,lambda:runtime.program_report().diagnostics)
        with collective_check(world):
            # The final h=.05 step must consume the retained [.1,.2] window.
            # Replacing its duration with either current dt or a label fails here.
            assert diagnostics["bdf2.history.begin"] == .1
            assert diagnostics["bdf2.history.duration"] == .1
    a=collective_call(world,lambda:uninterrupted.state_global("method"))
    b=collective_call(world,lambda:restarted.state_global("method"))
    _require_owned_array(world,a)
    _require_owned_array(world,b)
    with collective_check(world):
        assert uninterrupted.time()==restarted.time()==.25
        assert uninterrupted.macro_step()==restarted.macro_step()==3
        if a is not None:
            np.testing.assert_array_equal(a,b)
            assert np.max(abs(np.asarray(a)-saved_state)) > .05
    for name in ("bdf_state","bdf_increment"):
        left=collective_call(world,lambda:uninterrupted.history_global(name,1))
        right=collective_call(world,lambda:restarted.history_global(name,1))
        _require_owned_array(world,left)
        _require_owned_array(world,right)
        with collective_check(world):
            if left is not None:np.testing.assert_array_equal(left,right)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_t07_nonzero_finite_solve_defect_is_delivered_once_and_retained(
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile("T07",solve_error=.001)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=spec.t_end,max_steps=3,console=False))
    rows=profile_gather_value(world,collective_call(world,lambda:spec.invariants(runtime)))
    _assert_metrics(world,rows)
    with collective_check(world):
        assert runtime.time()==spec.t_end and runtime.macro_step()==3
    actual=collective_call(world,lambda:runtime.state_global("method"))
    _require_owned_array(world,actual)
    with collective_check(world):
        if actual is not None:
            # Independent scalar recurrence: first BE, then constant/variable
            # BDF2. Both accepted errors survive into the final state/history.
            u1=1/1.1
            u2=(2*u1-.5)/1.6+.001
            u3=(1.5*u2-u1/6)/(4/3+.05)+.001
            exact_u2=(2*u1-.5)/1.6
            exact_u3=(1.5*exact_u2-u1/6)/(4/3+.05)
            assert abs(u3-exact_u3)>.002
            target=np.array([u3,.05])[:,None,None]
            np.testing.assert_allclose(np.asarray(actual).reshape(2,4,4),
                np.broadcast_to(target,(2,4,4)),atol=2.e-12,rtol=0)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_t01_native_temporal_order_on_fixed_spatial_operator(isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    outputs=[]
    for dt in (.02,.01,.005,.00125):
        spec,artifact,world=_compile("T01",dt=dt,end=.04)
        runtime=_bind(spec,artifact,world)
        collective_call(world,lambda:pops.run(runtime,t_end=.04,max_steps=40,console=False))
        actual=collective_call(world,lambda:runtime.state_global("method"))
        _require_owned_array(world,actual)
        outputs.append(actual)
        rows=profile_gather_value(world,collective_call(world,lambda:spec.invariants(runtime)))
        _assert_metrics(world,rows)
        with collective_check(world):
            assert abs(runtime.time()-.04)<2.e-14
            assert runtime.macro_step()==round(.04/dt)
    with collective_check(world):
        if outputs[0] is not None:
            arrays=[np.asarray(value) for value in outputs]
            errors=[float(np.sqrt(np.mean((value-arrays[-1])**2))) for value in arrays[:-1]]
            ratios=[errors[i]/errors[i+1] for i in range(2)]
            assert min(errors)>1.e-12,errors
            assert np.isfinite(errors).all() and errors[0]>errors[1]>errors[2]
            assert min(ratios)>3.5,(errors,ratios)
            record_property("temporal_errors",str(errors));record_property("temporal_ratios",str(ratios))


@pytest.mark.compiler
@pytest.mark.native_loader
def test_t03_native_second_order_against_continuous_solution(
        isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    errors=[]
    exact=.2-1+2*math.exp(-.2)
    for dt in (.1,.05,.025):
        spec,artifact,world=_compile("T03",dt=dt,end=.2)
        runtime=_bind(spec,artifact,world)
        collective_call(world,lambda:pops.run(runtime,t_end=.2,max_steps=8,console=False))
        rows=profile_gather_value(world,collective_call(world,lambda:spec.invariants(runtime)))
        _assert_metrics(world,rows)
        actual=collective_call(world,lambda:runtime.state_global("method"))
        _require_owned_array(world,actual)
        with collective_check(world):
            assert abs(runtime.time()-.2)<2.e-14
            assert runtime.macro_step()==round(.2/dt)
            if actual is not None:
                state=np.asarray(actual).reshape(2,4,4)
                errors.append(float(np.max(abs(state[0]-exact))))
                np.testing.assert_allclose(state[1],.2,rtol=0,atol=2.e-14)
    with collective_check(world):
        if errors:
            assert len(errors)==3 and min(errors)>1.e-12
            ratios=[errors[i]/errors[i+1] for i in range(2)]
            # Strang's symmetric extended-system composition has global O(h^2)
            # error at fixed final time. A fixed 5% band around factor four is
            # declared before receipt; it is not fitted to native measurements.
            assert all(3.8<ratio<4.2 for ratio in ratios),(errors,ratios)
            record_property("continuous_solution_errors",str(errors))
            record_property("continuous_solution_error_ratios",str(ratios))


@pytest.mark.compiler
@pytest.mark.native_loader
def test_t08_composite_transport_improves_on_stationary_state(
        isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    from examples.api_v02.method_amr_reception import composite_base_projection, initial_base_means
    spec,artifact,world=_compile("T08")
    runtime=_bind(spec,artifact,world)
    cells=(32,32)
    def initial_projection():
        levels=runtime.n_levels()
        states=[runtime.block_level_state_global("method",level) for level in range(levels)]
        return composite_base_projection(states,runtime.patch_boxes(),cells=cells)
    projected=collective_call(world,initial_projection)
    before_boxes=tuple(runtime.patch_boxes())
    before=None if projected is None else projected["means"]
    _require_owned_array(world,before)
    with collective_check(world):
        # This is the assumption behind the continuous cell-average reference.
        # If native AMR initialization changes it, fail rather than calibrate the
        # expected answer against that changed initial state.
        assert runtime.time()==0. and runtime.macro_step()==0
        if before is not None:
            np.testing.assert_allclose(before,initial_base_means(cells),rtol=0,atol=2.e-13)
            counts=projected["active_counts"]
            assert counts[0]>0 and sum(counts[1:])>0,counts
    report=collective_call(world,lambda:pops.run(runtime,t_end=spec.t_end,max_steps=2,console=False))
    rows=profile_gather_value(world,collective_call(world,lambda:spec.invariants(runtime)))
    _assert_metrics(world,rows)
    with collective_check(world):
        assert runtime.macro_step()==2 and abs(runtime.time()-spec.t_end)<2.e-14
        assert report.accepted_steps==2 and report.rejected_steps==0
        assert tuple(runtime.patch_boxes())==before_boxes
        owners=[row for row in rows if row.get("metrics_owner",True)]
        for row in owners:
            assert row["has_active_coarse_fine_interface"]
            assert row["composite_state_change"]>row["comparison_roundoff"]
            assert (row["composite_translation_error"]+row["comparison_roundoff"]
                    <row["stationary_translation_error"])
        record_property("composite_translation_metrics",str(rows))


@pytest.mark.compiler
@pytest.mark.native_loader
def test_t08_requested_regrid_refuses_deep_subcycled_history_remap(
        isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile("T08",regrid=True)
    runtime=_bind(spec,artifact,world)
    _,failures=collective_attempt(world,lambda:pops.run(runtime,t_end=spec.t_end,max_steps=2,console=False))
    expected=("AMR Program deferred history remap supports initialized direct-child "
              "AB2 IntegralOnly 1:1 or 2:1 only; deeper state rings require exact 1:1 samples")
    with collective_check(world):
        assert failures and all(row and row[0]=="ValueError" and expected in row[1]
                                for row in failures),failures
    # A runtime regrid may be scheduled after a successful accepted step. Do
    # not assert rollback to the initial state or erase such an accepted step.
    # Record the actual surviving publication and its arrays as observations.
    after=collective_call(world,lambda:dict(time=runtime.time(),macro_step=runtime.macro_step(),
        levels=runtime.n_levels(),patch_boxes=runtime.patch_boxes()))
    for level in range(after["levels"]):
        actual=collective_call(world,lambda level=level:runtime.block_level_state_global("method",level))
        _require_owned_array(world,actual)
    record_property("accepted_window_after_unsupported_remap",str(after))
    record_property("rollback_to_initial_claimed",False)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_t08_parent_refusal_restores_entire_native_envelope(isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile("T08",reject_parent=True)
    runtime=_bind(spec,artifact,world)
    levels=runtime.n_levels()
    before=[collective_call(world,lambda level=level:runtime.block_level_state_global("method",level))
            for level in range(levels)]
    _require_owned_array(world,before[0])
    before_exchange=collective_call(world,lambda:bytes(runtime._executor.program_accepted_state()))
    before_report=collective_call(world,runtime.program_report)
    before_boxes=runtime.patch_boxes()
    _,failures=collective_attempt(world,lambda:pops.run(runtime,t_end=spec.dt,max_steps=1,console=False))
    after=[collective_call(world,lambda level=level:runtime.block_level_state_global("method",level))
           for level in range(levels)]
    after_exchange=collective_call(world,lambda:bytes(runtime._executor.program_accepted_state()))
    after_report=collective_call(world,runtime.program_report)
    with collective_check(world):
        assert failures and all(row and "parent_after_children" in row[1] for row in failures),failures
        assert runtime.time()==0. and runtime.macro_step()==0
        assert levels>=2 and runtime.n_levels()==levels
        assert runtime.patch_boxes()==before_boxes
        assert before_exchange and before_exchange==after_exchange
        assert before_report.histories==after_report.histories
        for a,b in zip(before,after,strict=True):
            if a is not None:
                assert np.asarray(a).size>0
                np.testing.assert_array_equal(a,b)
        if before[0] is not None:assert np.ptp(before[0])>.1


@pytest.mark.compiler
@pytest.mark.native_loader
def test_t09_interval_body_reexecutes_for_changed_duration(isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from examples.api_v02.methods_T01_T12 import build_t09_homogeneous
    spec,artifact,world=_compile("T09_homogeneous",builder=build_t09_homogeneous)
    results=[]
    for duration in (.1,.2):
        runtime=_bind(spec,artifact,world)
        collective_call(world,lambda:pops.run(runtime,t_end=duration,max_steps=1,console=False))
        actual=collective_call(world,lambda:runtime.state_global("method"))
        _require_owned_array(world,actual)
        with collective_check(world):
            assert runtime.time()==duration and runtime.macro_step()==1
            if actual is not None:
                actual=np.asarray(actual).reshape(2,4,4)
                factor=-math.expm1(-duration/.4);exchange=.3*.4*factor*1.5
                target=np.broadcast_to(np.array([2-factor-exchange,.5+.5*factor+exchange])[:,None,None],actual.shape)
                np.testing.assert_allclose(actual,target,atol=2.e-13,rtol=0)
                results.append(actual)
    with collective_check(world):
        if results:assert np.max(abs(results[0]-results[1]))>.01


@pytest.mark.compiler
@pytest.mark.native_loader
def test_t10_retry_uses_new_request_and_publishes_actual_time(isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile("T10",retry=True)
    runtime=_bind(spec,artifact,world)
    report=collective_call(world,lambda:pops.run(runtime,t_end=spec.t_end,max_steps=1,console=False))
    rows=profile_gather_value(world,collective_call(world,lambda:spec.invariants(runtime)))
    _assert_metrics(world,rows)
    with collective_check(world):
        assert report.rejected_steps==1 and report.accepted_steps==1
        assert abs(runtime.time()-8/17)<2.e-14
        assert runtime.macro_step()==1


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("initial",((1.,2.),(0.,2.),(1.,0.),(0.,0.)))
def test_t11_equal_net_ode_distinct_directed_schemes_and_empty_species(
        initial,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    outputs=[]
    for circulation in (0.,.7):
        spec,artifact,world=_compile("T11",circulation=circulation,initial=initial)
        runtime=_bind(spec,artifact,world)
        collective_call(world,lambda:pops.run(runtime,t_end=spec.t_end,max_steps=2,console=False))
        actual=collective_call(world,lambda:runtime.state_global("method"))
        _require_owned_array(world,actual)
        outputs.append(actual)
        rows=profile_gather_value(world,collective_call(world,lambda:spec.invariants(runtime)))
        _assert_metrics(world,rows)
        with collective_check(world):
            assert runtime.time()==spec.t_end and runtime.macro_step()==2
    with collective_check(world):
        x,y=initial
        rates=[-(.3*x+c*x*y)+(.2*y+c*x*y) for c in (0.,.7)]
        assert abs(rates[0]-rates[1])<1.e-14
        if outputs[0] is not None and initial==(1.,2.):
            assert np.max(abs(np.asarray(outputs[0])-outputs[1]))>1.e-3


@pytest.mark.parametrize("method", (
    "T01", "T02", "T03", "T04", "T05", "T06",
    "T07", "T08", "T09", "T10", "T11", "T12",
))
def test_unqualified_family_obligations_are_explicit(method):
    with pytest.raises(MissingCapabilityError) as failure:
        build_method(method,require_complete=True)
    assert failure.value.required == OBLIGATIONS[method]
    assert failure.value.required


def test_unknown_method_does_not_fall_back():
    with pytest.raises(ValueError,match="unknown method"):
        build_method("T99")
