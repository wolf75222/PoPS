"""Authored temporal AMR transfer and a nonuniform BDF3 consumer.

No interpolation or BDF formula below has a native method-name implementation.
The native runtime applies the generated scalar matrix to immutable numeric and
flux histories, then the ordinary Program consumes its authenticated intervals.
"""
import numpy as np
import pops
from pops import math as pm
from pops.time import HistoryTransfer, FixedDt, FailRun, RejectAttempt, every
from pops.time.expressions import ProgramScalar
from .common import CaseSpec, non_owner_metrics
from .models_M01_M14 import _domain, _initial, _fv


def quadratic_transfer(*, linear_control=False, reject=False):
    def equations(samples, accepted_time, ratio):
        times = tuple(sample.time for sample in samples)
        # Slot zero is recycled. The two readable lags are made strictly
        # ordered, with an explicitly nonuniform newest interval. No invented
        # recent sample is extrapolated between the parent and accepted state.
        targets = (times[0], times[1], (times[1]+times[2])/2)
        intervals = (samples[0].interval, accepted_time-targets[1], targets[1]-targets[2])
        valid = (ratio > 0) & (times[1] > times[2]) & (times[2] > times[0])
        if reject:
            valid = valid & (ratio < 0)
        def weights(target):
            if linear_control:
                return (0., (target-times[2])/(times[1]-times[2]),
                        (times[1]-target)/(times[1]-times[2]))
            return tuple(_product((target-times[j])/(times[i]-times[j])
                                  for j in range(3) if j != i) for i in range(3))
        return valid, tuple((target, interval, weights(target))
                            for target, interval in zip(targets, intervals))
    return HistoryTransfer(slots=3, body=equations)


def _product(values):
    result = 1
    for value in values:
        result = result*value
    return result


def _derivative_coefficients(nodes):
    """Derivative at nodes[0] of each Lagrange basis, entirely Python algebra."""
    return tuple(sum(_product(nodes[0]-nodes[k] for k in range(len(nodes))
                              if k not in (i,j))
                     for j in range(len(nodes)) if j != i)
                 /_product(nodes[i]-nodes[k] for k in range(len(nodes)) if k != i)
                 for i in range(len(nodes)))


def build_history_transfer(cells=(16,16), *, linear_control=False,
                           reject_transfer=False, reject_parent=False):
    from pops.amr import (AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid,
        AMRTagging, AMRTransfer, Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, StateStorage
    frame, _ = _domain("T08_history_transfer", cells)
    model = pops.Model("manufactured_clock_polynomial", frame=frame)
    U = model.state("u", components=("quadratic", "clock", "tag"))
    source = model.source("clock_polynomial_source", on=U, value=(2*U[1],1.,1.))
    rate = model.rate("balance", equation=pm.ddt(U)==source)
    case = pops.Case("T08_authored_history_transfer")
    block = case.block("method", model)
    # This manufactured equation is local. The separate flux-history receiver
    # exercises genuine transport/reflux; a zero transport port would impose a
    # spatial ledger on an algebraic BDF solve that has no spatial contribution.
    plan = DiscretizationPlan(); plan.rates.add(rate, StateStorage()); case.numerics(plan,block=block)
    p = pops.Program("nonuniform_BDF3_after_AMR_transfer"); q = p.state(block[U])
    p.keep_history(q, depth=2, interpolation=quadratic_transfer(
        linear_control=linear_control, reject=reject_transfer))
    h = ProgramScalar(p.requested_dt())
    def bdf(builder):
        previous = (q.prev(1).value, q.prev(2).value)
        bound = []
        for lag, old in enumerate(previous,1):
            interval = builder.history_interval(old)
            state = interval.bind(builder,old,name="lag_%d" % lag)
            mismatch = builder.value("lag_clock_error_%d" % lag,
                                     (abs(state[1]-interval.begin),)*3)
            state = builder.guard("authenticated_lag_%d" % lag,state,
                                  builder.max(mismatch)<1.e-12,action=FailRun())
            builder.record_scalar("transfer.lag_%d.begin" % lag,interval.begin)
            builder.record_scalar("transfer.lag_%d.duration" % lag,interval.duration)
            bound.append(state)
        target = q.n[1]+h
        coefficients = _derivative_coefficients((target,q.n[1],bound[0][1],bound[1][1]))
        known = (q.n[0],bound[0][0],bound[1][0])
        solution = (2*target-sum(coefficients[i+1]*known[i] for i in range(3)))/coefficients[0]
        return builder.value("BDF3_all_lags",(solution,target,q.n[2]+h),at=q.next.point)
    def startup(builder):
        k0 = rate(q.n)
        predictor = builder.value("startup_predictor",q.n+p.dt*k0,at=q.next.point)
        return builder.value("startup_Heun",q.n+.5*p.dt*(k0+rate(predictor)),at=q.next.point)
    clock = p.value("clock_for_startup",(q.n[1],)*3)
    candidate = p.branch(p.min(clock)>0.06,bdf,startup,name="two_step_startup_or_BDF3")
    p.commit(q.next,candidate)
    if reject_parent:
        def after_sync(builder):
            state = builder.value("synchronized",q.n,at=q.next.point)
            clock = builder.value("synchronized_clock",(state[1],)*3)
            checked = builder.guard("reject_after_child_work",state,
                                    builder.max(clock)<.161,action=RejectAttempt())
            builder.commit(q.next,checked)
        p.after_synchronization(after_sync)
    p.step_strategy(FixedDt(.04)); case.program(p)
    subject = block[U]
    transfer = AMRTransfer(); transfer.state(subject,StateTransfer())
    from pops.params import RuntimeParam
    threshold = case.param(RuntimeParam("refine_threshold", default=1.))
    layout = AMR(grid=CartesianGrid(frame=frame,cells=cells,periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2,ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(pm.ValueExpr(subject)[2]>case.value(threshold)),Buffer(cells=1)),
            hysteresis=Hysteresis(0,EqualityPolicy.HOLD),conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(4,clock=p.clock)),transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0,1,2),)))
    x,y=np.meshgrid((np.arange(cells[0])+.5)/cells[0],(np.arange(cells[1])+.5)/cells[1])
    tag=np.where((abs(x-.5)<.15)&(abs(y-.5)<.15),1.2,.9)
    initial=np.stack((0*x,0*x,tag))
    def check(runtime):
        levels=runtime.n_levels()
        states=[runtime.block_level_state_global("method",level) for level in range(levels)]
        if states[0] is None:return non_owner_metrics()
        errors=[]
        # History checks below use sparse geometry; active state checks ignore
        # dense holes explicitly using the live patch layout.
        for level,state in enumerate(states):
            values=np.asarray(state).reshape(3,cells[1]*2**level,cells[0]*2**level)
            finite=values[1]>0
            if np.any(finite):
                errors.append(float(np.max(abs(values[0][finite]-values[1][finite]**2))))
                errors.append(float(np.max(abs(values[1][finite]-runtime.time()))))
        error=max(errors,default=float("inf"))
        return dict(passed=error<2.e-11,metrics_owner=True,quadratic_error=error,levels=levels)
    return CaseSpec(case,layout,_initial(case,subject,initial),{"method":subject},.18,.04,check,
        "T08: local source BDF3 with generated three-slot quadratic AMR history transfer; 2:1 clocks, actual new coverage, "
        "nonuniform reconstructed lag intervals, BDF3 consuming every readable lag, final shortened step")


def regular_quadratic_flux_transfer():
    """An explicitly extrapolating quadratic reconstruction onto child lag times."""
    def equations(samples,accepted_time,ratio):
        times=tuple(sample.time for sample in samples)
        h=samples[1].interval/ratio
        targets=(accepted_time-3*h,accepted_time-h,accepted_time-2*h)
        valid=(ratio>0)&(times[1]>times[2])&(times[2]>times[0])
        return valid,tuple((target,h,tuple(_product((target-times[j])/(times[i]-times[j])
            for j in range(3) if j!=i) for i in range(3))) for target in targets)
    return HistoryTransfer(slots=3,body=equations)


def build_flux_history_transfer(cells=(32,32)):
    """Real spatial flux history, fixed-step AB3 and generated quadratic transfer.

    F=t²*reference has time-quadratic face samples on each fixed hierarchy.
    Source histories therefore give an independent temporal scaling oracle,
    including when distinct accepted times recycled the same attempt-local ID.
    """
    from fractions import Fraction as F
    from pops.amr import (AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid,
        AMRTagging, AMRTransfer, Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan
    from pops.params import RuntimeParam
    frame,_=_domain("T08_flux_history_transfer",cells)
    model=pops.Model("time_polynomial_flux",frame=frame)
    U=model.state("u",components=("transported","clock","tag","reference"))
    flux=model.flux("time_quadratic_flux",state=U,frame=frame,
        components={frame.axes[0]:(U[1]**2*U[3],0.,0.,0.),frame.axes[1]:(0.,0.,0.,0.)},
        waves={axis:(0.,0.,0.,0.) for axis in frame.axes})
    source=model.source("clock_and_tag",on=U,value=(0.,1.,1.,0.))
    rate=model.rate("balance",equation=pm.ddt(U)==-pm.div(flux)+source)
    case=pops.Case("T08_flux_history");block=case.block("method",model)
    plan=DiscretizationPlan();plan.rates.add(rate,_fv(flux,U));case.numerics(plan,block=block)
    p=pops.Program("AB3_with_authored_flux_history_transfer");q=p.state(block[U])
    current=rate(q.n)
    p.store_history("retained_rhs",current,depth=2,interpolation=regular_quadratic_flux_transfer())
    def ab3(builder):
        old=[]
        for lag in (1,2):
            value=builder.history("retained_rhs",lag,space=q.n.space,block=q.n.block,state_ref=q.n.state_ref)
            interval=builder.history_interval(value)
            old.append(interval.bind(builder,value,name="authentic_rhs_%d" % lag))
        return builder.value("AB3",q.n+p.dt*(F(23,12)*current-F(16,12)*old[0]+F(5,12)*old[1]),at=q.next.point)
    def startup(builder):
        midpoint=builder.stage("startup_midpoint",c=F(1,2))
        middle=builder.value("middle",q.n+F(1,2)*p.dt*current,at=midpoint)
        rm=rate(middle)
        last=builder.value("last",q.n-p.dt*current+2*p.dt*rm,at=q.next.point)
        return builder.value("RK3_startup",q.n+p.dt*(F(1,6)*current+F(2,3)*rm+F(1,6)*rate(last)),at=q.next.point)
    clock=p.value("clock",(q.n[1],)*4)
    candidate=p.branch(p.min(clock)>.06,ab3,startup,name="mature_AB3_or_RK3")
    p.commit(q.next,candidate);p.step_strategy(FixedDt(.04));case.program(p)
    subject=block[U];transfer=AMRTransfer();transfer.state(subject,StateTransfer())
    threshold=case.param(RuntimeParam("refine_threshold",default=1.))
    layout=AMR(grid=CartesianGrid(frame=frame,cells=cells,periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2,ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(pm.ValueExpr(subject)[2]>case.value(threshold)),Buffer(cells=1)),
            hysteresis=Hysteresis(0,EqualityPolicy.HOLD),conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(4,clock=p.clock)),transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0,1,2),)))
    x,y=np.meshgrid((np.arange(cells[0])+.5)/cells[0],(np.arange(cells[1])+.5)/cells[1])
    central=(abs(x-.45)<.12)&(abs(y-.5)<.12)
    new_region=(x>.6)&(x<.75)&(abs(y-.5)<.12)
    later_region=(x>.8)&(x<.9)&(abs(y-.5)<.12)
    tag=np.where(central,1.2,np.where(new_region,.9,np.where(later_region,.7,.6)))
    initial=np.stack((1+0*x,0*x,tag,np.sinc(1/cells[0])*np.sin(2*np.pi*x)))
    def check(runtime):
        mass=runtime.integral("method",component=0)
        return dict(passed=abs(mass-1.)<3.e-11,metrics_owner=True,mass_defect=abs(mass-1.))
    return CaseSpec(case,layout,_initial(case,subject,initial),{"method":subject},.36,.04,check,
        "T08: real time-quadratic spatial face flux, RK3 startup then AB3; three RHS slots reconstructed "
        "through Python-authored matrix, active coarse/fine interface retained, reflux and strict restart")
