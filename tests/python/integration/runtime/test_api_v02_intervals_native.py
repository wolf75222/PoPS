"""Native Interval/HistoryInterval tests; historical authority requires build5."""
import numpy as np
import pops
import pytest

from examples.api_v02.methods_T01_T12 import _storage, _finish
from pops.contributions import Contribution, Delivery, Increment
from pops.time import ComputedDt, RejectAttempt
from pops.time.expressions import ProgramExpression, ProgramScalar
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile, profile_gather_value


def _build(*, wrong=False, retry=False):
    parts=_storage("issued_interval",("u","elapsed"),(0.,0.))
    *_,p,q,_initial=parts
    h=p.requested_dt()
    window=p.interval(duration=h*2 if wrong else h)
    amount=p.value("equation",ProgramExpression((ProgramScalar(h),ProgramScalar(h)),q.n),at=q.next.point)
    item=Contribution("J",amount,window)
    increment=Increment((item,),(Delivery(item,q.n,((1,0),(0,1))),))
    result=increment.apply(p,q.n,name="native_interval_delivery",at=q.next.point)
    strategy=None
    if retry:
        result=p.guard("reject_first_interval",result,h<.075,action=RejectAttempt())
        p.reached_duration(h)
        strategy=ComputedDt(.1,max_rejections=1,endpoint_ulps=4)
    return _finish(parts,result,(.05,.05) if retry else (.1,.1),dt=.1,
        end=.05 if retry else .1,strategy=strategy,profile="issued root Interval native check")


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("wrong,retry",((True,False),(False,True)))
def test_runtime_interval_mismatch_or_fresh_retry(wrong,retry,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope="issued runtime interval")
    spec=collective_call(world,lambda:_build(wrong=wrong,retry=retry))
    resolved=collective_call(world,lambda:pops.resolve(pops.validate(spec.case),layout=spec.layout))
    artifact=compile_profile_plan_once(world,resolved,route="issued Interval",compile_artifact=pops.compile)
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values=spec.initial_values,
        resources={"execution_context":artifact_execution_context(artifact)}))
    before=collective_call(world,lambda:runtime.state_global("method"))
    # ComputedDt deliberately proposes .1 before accepting the retried .05.
    report,failures=collective_attempt(world,lambda:pops.run(runtime,t_end=spec.t_end,max_steps=1,console=False))
    after=collective_call(world,lambda:runtime.state_global("method"))
    owners=profile_gather_value(world,after is not None)
    with collective_check(world):
        assert any(owners)
        if wrong:
            assert failures and all(f and "native_interval_delivery_interval" in f[1] for f in failures)
            assert runtime.time()==0 and runtime.macro_step()==0
            if after is not None:np.testing.assert_array_equal(before,after)
        else:
            assert not any(failures),failures
            assert report.rejected_steps==1 and report.accepted_steps==1
            assert runtime.time()==.05 and runtime.macro_step()==1
            if after is not None:np.testing.assert_allclose(after,.05,rtol=0,atol=2.e-15)


def _build_history(*, missing=False):
    parts=_storage("historical_interval",("u","ready"),(0.,0.))
    *_,p,q,_initial=parts
    window=p.interval()
    def retained(b):
        value=b.history("retained",lag=1,space=q.n.space,block=q.n.block,state_ref=q.n.state_ref)
        past=b.history_interval(value)
        item=Contribution("past",value,past)
        old=Increment((item,),(Delivery(item,q.n,((1,0),(0,0))),))
        if not missing:
            # Capture precedes a new publication in the same retained ring.
            # The exact lifecycle snapshot must no longer be accepted.
            b.store_history("retained",q.n,depth=1)
        weighted=old.scale(b,1.,name="weighted_past",interval=window)
        return weighted.apply(b,q.n,name="history_delivery",at=q.next.point)
    def startup(b):
        b.store_history("retained",q.n,depth=1)
        return b.value("startup",ProgramExpression((.1,1.),q.n),at=q.next.point)
    candidate=retained(p) if missing else p.branch(p.max(p.value("ready",(q.n[1],q.n[1])))>0,
        retained,startup,name="startup_or_history")
    return _finish(parts,candidate,(.1,1.),dt=.1,end=.2,
        profile="adverse retained publication authority, no numerical acceptance intended")


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("missing",(True,False))
def test_history_interval_read_or_lifecycle_failure_rolls_back(missing,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope="retained historical interval")
    spec=collective_call(world,lambda:_build_history(missing=missing))
    resolved=collective_call(world,lambda:pops.resolve(pops.validate(spec.case),layout=spec.layout))
    artifact=compile_profile_plan_once(world,resolved,route="HistoryInterval",compile_artifact=pops.compile)
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values=spec.initial_values,
        resources={"execution_context":artifact_execution_context(artifact)}))
    if not missing:
        collective_call(world,lambda:pops.run(runtime,t_end=.1,max_steps=1,console=False))
    before=collective_call(world,lambda:runtime.state_global("method"))
    old_time,old_step=runtime.time(),runtime.macro_step()
    saved=None if missing else collective_call(world,lambda:runtime.history_global("retained",1))
    _,failures=collective_attempt(world,lambda:pops.run(runtime,t_end=.2,max_steps=1,console=False))
    after=collective_call(world,lambda:runtime.state_global("method"))
    owners=profile_gather_value(world,after is not None)
    with collective_check(world):
        assert any(owners)
        assert failures and all(f for f in failures),failures
        if not missing:
            assert all("weighted_past_origin_0_history_origin" in f[1] for f in failures),failures
        assert runtime.time()==old_time and runtime.macro_step()==old_step
        if after is not None:np.testing.assert_array_equal(before,after)
    if not missing:
        restored=collective_call(world,lambda:runtime.history_global("retained",1))
        with collective_check(world):
            if restored is not None:np.testing.assert_array_equal(saved,restored)
