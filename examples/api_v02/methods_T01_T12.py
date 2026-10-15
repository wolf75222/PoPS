"""Native Program constructions for twelve precisely bounded method profiles.

All step, residual, iteration and quadrature bodies below author PoPS IR. Python
and NumPy only prepare initial arrays and independently assess native results.
These finite profiles are not qualifications of every PDE in the T01--T12 corpus.
``LIMITS`` records scientific properties still unestablished; ``require_complete``
checks selected equation-component coverage separately. No custom emitter or
model-name dispatch is registered in the engine.
"""
from __future__ import annotations

from fractions import Fraction as F
import math as reference_math
import numpy as np
import pops
from pops import math as pm
from pops.contributions import Contribution, Delivery, Increment, bdf2_increment
from pops.realization import Realize, Use
from pops.time import (ComputedDt, FailRun, FixedDt, LocalResidual, RejectAttempt)
from pops.time.expressions import ProgramExpression, ProgramScalar
from pops.solvers import LocalNewton

from .common import CaseSpec, non_owner_metrics, state_array
from .coverage import OBLIGATIONS
from .models_M01_M14 import _domain, _initial, _fv, _poisson, _live_field_evaluator


READY = tuple(f"T{i:02d}" for i in range(1, 13))
# Compatibility surface; coverage is the single source of remaining obligations.
LIMITS = {identifier: OBLIGATIONS[identifier] for identifier in READY}


def _storage(name, components, initial, *, cells=(4, 4), configure=None, sampling=None):
    frame, layout = _domain(name, cells)
    model = pops.Model(name, frame=frame)
    state = model.state("q", components=components, **({} if sampling is None else {"sampling":sampling}))
    if configure is not None:
        configure(model,state)
    case = pops.Case(name)
    block = case.block("method", model, states=(state,))
    program = pops.Program(name + "_equations")
    temporal = program.state(block[state])
    initial = np.broadcast_to(np.asarray(initial, dtype=float)[:, None, None],
                              (len(components), cells[1], cells[0])).copy()
    return case, layout, model, block, state, program, temporal, initial


def _finish(parts, candidate, expected, *, dt, end=None, profile, tolerance=2.e-10,
            strategy=None, extra=None):
    case, layout, model, block, state, program, temporal, initial = parts
    program.commit(temporal.next, candidate)
    program.step_strategy(FixedDt(dt) if strategy is None else strategy)
    case.program(program)
    subject = block[state]
    values = _initial(case, subject, initial)
    expected = np.broadcast_to(np.asarray(expected)[:, None, None], initial.shape)

    def check(runtime):
        actual = state_array(runtime, "method", initial.shape)
        if actual is None:
            # Extra callbacks may themselves contain collective gathers.
            if extra is not None:
                extra(runtime, None)
            return non_owner_metrics()
        error = float(np.max(np.abs(actual - expected)))
        result = dict(passed=bool(np.all(np.isfinite(actual)) and error < tolerance),
                      metrics_owner=True, reference_error=error)
        if extra is not None:
            additions = extra(runtime, actual)
            result["passed"] = bool(result["passed"] and additions.pop("passed", True))
            result.update(additions)
        return result

    return CaseSpec(case, layout, values, {"method": subject},
                    dt if end is None else end, dt, check, profile)


def build_t01(cells=(8, 8), *, dt=.001, end=None):
    """SSPRK2 with three distinct live Poisson observations, including final state."""
    from pops.numerics import DiscretizationPlan
    lengths = (2*np.pi, 2*np.pi)
    frame, layout = _domain("T01_stage_fields", cells, lengths=lengths)
    model = pops.Model("T01_guiding_center", frame=frame)
    U = model.state("density", components=("q",)); phi = model.field("potential")
    gradient = model.vector("gradient", frame=frame,
        components={frame.axes[0]: pm.grad(phi).x, frame.axes[1]: pm.grad(phi).y})
    flux = model.flux("transport", state=U, frame=frame,
        components={frame.axes[0]: (-gradient.y*U[0],), frame.axes[1]: (gradient.x*U[0],)},
        waves={frame.axes[0]: (-gradient.y,), frame.axes[1]: (gradient.x,)})
    rate = model.rate("balance", equation=pm.ddt(U) == -pm.div(flux))
    case = pops.Case("T01_SSPRK2_fields"); block = case.block("method", model)
    plan = DiscretizationPlan(); plan.rates.add(rate, _fv(flux, U)); case.numerics(plan, block=block)
    field = _poisson(case, phi, U[0])
    if not 0 < dt <= .02:raise ValueError("T01 reception requires 0 < dt <= .02")
    end=2*dt if end is None else end
    if end < dt:raise ValueError("T01 reception must include at least one complete step")
    p = pops.Program("T01_explicit_stages"); q = p.state(block[U])
    evaluate = _live_field_evaluator(field, phi, model, block, U, rate)
    point0 = p.stage("begin", c=0)
    k0 = evaluate(p, q.n, point0, "stage0")
    point1 = p.stage("predictor", c=1)
    q1 = p.value("predictor", q.n + p.dt*k0, at=point1)
    k1 = evaluate(p, q1, point1, "stage1")
    candidate = p.value("accepted", .5*q.n + .5*(q1+p.dt*k1), at=q.next.point)
    # Final field is explicitly recomputed on the accepted candidate. The last
    # stage's field has a different state version despite the same physical time.
    final = field.observe(p.solve(field, values={block[U]: candidate},
                                  at=q.next.point).consume(action=FailRun()))
    p.store_history("final_state", candidate, depth=1)
    p.store_history("final_potential", final[field[phi]], depth=1)
    p.commit(q.next, candidate); p.step_strategy(FixedDt(dt)); case.program(p)
    x, y = np.meshgrid(2*np.pi*(np.arange(cells[0])+.5)/cells[0],
                       2*np.pi*(np.arange(cells[1])+.5)/cells[1])
    initial = (np.sinc(1/cells[0])*np.sin(x) + .3*np.sinc(2/cells[1])*np.cos(2*y))[None]
    def check(runtime):
        actual = state_array(runtime, "method", initial.shape)
        histories = {label: (runtime.history_global(label+"_state", 1),
                             runtime.history_global(label+"_potential", 1))
                     for label in ("stage0", "stage1", "final")}
        if actual is None:
            return non_owner_metrics()
        residuals = {}
        for label, (u, potential) in histories.items():
            u = np.asarray(u).reshape(initial.shape)[0]
            potential = np.asarray(potential).reshape(initial.shape[1:])
            positive_lap = sum((2*potential-np.roll(potential, 1, axis)-np.roll(potential, -1, axis))
                /(lengths[1-axis]/cells[1-axis])**2 for axis in (0, 1))
            residuals[label+"_residual"] = float(np.max(abs(positive_lap-u)))
        final_error = float(np.max(abs(np.asarray(histories["final"][0]).reshape(initial.shape)-actual)))
        difference = float(np.max(abs(np.asarray(histories["stage1"][0])-histories["final"][0])))
        return dict(passed=max(residuals.values()) < 2.e-8 and final_error < 1.e-13
                    and difference > 1.e-12, metrics_owner=True, final_state_error=final_error,
                    final_vs_last_stage=difference, **residuals)
    return CaseSpec(case, layout, _initial(case, block[U], initial), {"method": block[U]},
                    end, dt, check, "T01: explicit SSPRK2 on periodic guiding-center PDE; fresh stage0, stage1 and final Poisson solves with separately retained native residual inputs")


def build_t02(cells=(8, 8)):
    """Explicit local decay and globally implicit periodic diffusion."""
    from pops.numerics import Diffusion, DiscretizationPlan, StateStorage
    from pops.solvers import Newton
    from pops.time import ImplicitStage, SolveUnknown, DerivativeStrategy
    frame, layout = _domain("T02_global_IMEX", cells)
    model = pops.Model("T02_diffusion_decay", frame=frame); U = model.state("u", components=("u",))
    source = model.source("decay", on=U, value=(-.7*U[0],))
    flux = model.diffusive_flux("conduction", state=U, value=pm.CoeffGradient(U[0], .04))
    balance = model.rate("balance", equation=pm.ddt(U) == source + pm.div(flux))
    explicit = balance.select(balance.occurrences[0])
    implicit = balance.select(balance.occurrences[1])
    case = pops.Case("T02_IMEX"); block = case.block("method", model)
    plan = DiscretizationPlan(); plan.rates.add(explicit, StateStorage()); plan.rates.add(implicit, Diffusion(flux=flux))
    case.numerics(plan, block=block)
    p = pops.Program("T02_IMEX_equations"); q = p.state(block[U]); dt = .01
    predictor = p.value("explicit", q.n + p.dt*explicit(q.n), at=q.next.point)
    request = ImplicitStage(implicit, predictor, p.dt).request(
        unknown=SolveUnknown("next_u", predictor), seed=predictor,
        derivative=DerivativeStrategy("finite_difference"))
    candidate = p.solve(request, solver=Newton(tolerance=1.e-12, max_iterations=12,
        linear_tolerance=1.e-11, linear_max_iterations=120, restart=30)).consume(action=FailRun())[0]
    candidate = p.value("accepted", candidate, at=q.next.point)
    p.commit(q.next, candidate); p.step_strategy(FixedDt(dt)); case.program(p)
    x, y = np.meshgrid((np.arange(cells[0])+.5)/cells[0], (np.arange(cells[1])+.5)/cells[1])
    wave = .2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
    initial = (1+wave)[None]
    eigenvalue = -4*.04*sum(n*n*np.sin(np.pi/n)**2 for n in cells)
    expected = ((1-.7*dt)**2*(1 + wave/(1-dt*eigenvalue)**2))[None]
    def check(runtime):
        actual = state_array(runtime, "method", initial.shape)
        if actual is None: return non_owner_metrics()
        error = float(np.max(abs(actual-expected)))
        return dict(passed=error < 2.e-9, metrics_owner=True, discrete_fourier_error=error)
    return CaseSpec(case, layout, _initial(case, block[U], initial), {"method":block[U]},
                    2*dt, dt, check, "T02: two IMEX-Euler steps, explicit cell-local decay plus globally coupled backward-Euler diffusion; independent discrete Fourier inverse")


def build_t03(*, dt=.1, end=.2):
    # Exact subflows: A=(du/dt=t, dt/dt=1), B=(du/dt=-u, dt/dt=0).
    dt, end = float(dt), float(end)
    if not reference_math.isfinite(dt) or not reference_math.isfinite(end) or dt <= 0 or end <= 0:
        raise ValueError("T03 reception requires positive finite dt and end")
    count = end/dt
    if not reference_math.isfinite(count) or count < 1:
        raise ValueError("T03 reception requires end=N*dt with positive integer N")
    steps = round(count)
    if not reference_math.isclose(steps*dt,end,rel_tol=1.e-13,abs_tol=0.):
        raise ValueError("T03 reception requires end=N*dt with positive integer N")
    parts = _storage("T03_extended_Strang", ("u", "physical_time"), (1., 0.))
    *_, p, q, _initial_values = parts
    h = ProgramScalar(p.requested_dt()); midpoint = p.stage("A_half", c=F(1,2))
    first = p.value("first_A", (q.n[0]+q.n[1]*h/2+h*h/8, q.n[1]+h/2), at=midpoint)
    middle = p.value("B_clock_frozen", (pm.exp(-h)*first[0], first[1]), at=midpoint)
    candidate = p.value("second_A", (middle[0]+middle[1]*h/2+h*h/8, middle[1]+h/2), at=q.next.point)
    value,time=1.,0.
    for _ in range(steps):
        value = reference_math.exp(-dt)*(value+time*dt/2+dt*dt/8)+(time+dt/2)*dt/2+dt*dt/8
        time += dt
    # Independent continuous solution of the combined extended equation u'=t-u.
    exact = end-1+2*reference_math.exp(-end)
    def check(runtime,actual):
        if actual is None:return {}
        clock_error=float(np.max(abs(actual[1]-runtime.time())))
        continuous_error=float(np.max(abs(actual[0]-exact)))
        return dict(passed=clock_error<2.e-14,actual_clock_error=clock_error,
                    continuous_solution_error=continuous_error)
    return _finish(parts, candidate, (value,time), dt=dt, end=end, extra=check,
        profile="T03: nonautonomous extended-system Strang, exact A/B subflows and independent continuous solution; A advances t twice by h/2 and B freezes t; no spatial boundary lifting qualified")


def build_t04():
    """Same finite block relation solved by a local Newton solve and Schur algebra."""
    parts = _storage("T04_Schur", ("x", "y", "f", "g"), (0.,0.,1.,2.))
    *_, p, q, _initial_values = parts
    seed = p.value("seed", q.n, at=q.next.point)
    def residual(_p, unknown, old):
        return (2*unknown[0]+.4*unknown[1]-old[2], .3*unknown[0]+3*unknown[1]-old[3],
                unknown[2]-old[2], unknown[3]-old[3])
    monolithic = p.solve(LocalResidual(residual, seed, captures={"old":q.n}),
        solver=LocalNewton(tolerance=1.e-13, max_iterations=10)).consume(action=FailRun())
    y = (q.n[3]-.3*q.n[2]/2)/(3-.3*.4/2)
    candidate = p.value("Schur_back_substitution", ((q.n[2]-.4*y)/2, y, q.n[2], q.n[3]), at=q.next.point)
    p.store_history("monolithic", monolithic, depth=1)
    expected = np.linalg.solve(np.array([[2,.4],[.3,3.]]), np.array([1.,2.]))
    def compare(runtime, actual):
        mono = runtime.history_global("monolithic", 1)
        if actual is None: return {}
        error = float(np.max(abs(np.asarray(mono).reshape(actual.shape)-actual)))
        return dict(passed=error < 2.e-11, monolithic_Schur_error=error)
    return _finish(parts, candidate, (*expected,1.,2.), dt=.1, end=.2, extra=compare,
        profile="T04: cell-local 2x2 nonsymmetric block relation, native monolithic LocalResidual versus explicitly authored Schur complement/back substitution; the T04 dispatcher selects the separately authored global Schur/CG profile")


def build_t04_global():
    """Monolithic and Schur CG of the same actual periodic elliptic block matrix."""
    from .method_krylov import conjugate_gradient, laplacian
    parts=_storage("T04_global_Schur",("x","y"),(0.,0.),cells=(8,8),sampling="cell_average")
    case,layout,model,block,U,p,q,initial=parts
    x,y=np.meshgrid((np.arange(8)+.5)/8,(np.arange(8)+.5)/8)
    initial[0]=1+.2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
    initial[1]=2+.3*np.sin(2*np.pi*x)
    rhs=p.value("original_rhs",q.n,at=q.next.point)
    def original(builder,value):
        L=laplacian(builder,value,name="original_laplacian")
        return builder.value("original_A",(-L[0]+2*value[0]+.4*value[1],
                                             .4*value[0]+3*value[1]),at=value.point)
    monolithic=conjugate_gradient(p,rhs,original,name="monolithic",max_steps=100)
    reduced_rhs=p.value("Schur_rhs",(rhs[0]-.4*rhs[1]/3,0.),at=q.next.point)
    def reduced(builder,value):
        L=laplacian(builder,value,name="Schur_laplacian")
        return builder.value("Schur_A",(-L[0]+(2-.16/3)*value[0],value[1]),at=value.point)
    reduced_solution=conjugate_gradient(p,reduced_rhs,reduced,name="condensed",max_steps=100)
    candidate=p.value("back_substitution",(reduced_solution[0],
        (rhs[1]-.4*reduced_solution[0])/3),at=q.next.point)
    p.store_history("monolithic",monolithic,depth=1)
    p.commit(q.next,candidate);p.step_strategy(FixedDt(.1));case.program(p)
    expected=initial.copy()
    frequency=np.arange(8);eigen=4*8**2*np.sin(np.pi*frequency/8)**2
    # Independent dense Fourier solves use the exact same finite periodic symbol.
    for _ in range(2):
        transformed=np.fft.fft2(expected,axes=(-2,-1))
        for j in range(8):
            for i in range(8):
                transformed[:,j,i]=np.linalg.solve(
                    np.array([[eigen[i]+eigen[j]+2,.4],[.4,3.]]),transformed[:,j,i])
        expected=np.fft.ifft2(transformed,axes=(-2,-1)).real
    def check(runtime):
        actual=state_array(runtime,"method",initial.shape)
        raw_mono=runtime.history_global("monolithic",1)
        if actual is None:return non_owner_metrics()
        mono=np.asarray(raw_mono).reshape(actual.shape)
        discrepancy=float(np.max(abs(actual-mono)));error=float(np.max(abs(actual-expected)))
        return dict(passed=discrepancy<2.e-10 and error<2.e-10,metrics_owner=True,
                    monolithic_Schur_error=discrepancy,independent_Fourier_error=error)
    return CaseSpec(case,layout,_initial(case,block[U],initial),{"method":block[U]},.2,.1,check,
        "T04: same global periodic A=[[-Delta+2,.4],[.4,3]] solved monolithically and by Schur with CG authored entirely through product/iterate/collective reductions; independent Fourier block inverse, no equation-specific emitter")


def _entropy_closure(p, moments, *, label):
    """Newton of sum exp(alpha0+v alpha1)*(1,v) at v=(-1,+1), weights 1."""
    zero = p.value(label+"_seed", ProgramExpression((0.,0.), moments), at=moments.point)
    carried = p.product(alpha=zero, target=moments)
    def residual(builder, state):
        alpha, target = state["alpha"], state["target"]
        minus, plus = pm.exp(alpha[0]-alpha[1]), pm.exp(alpha[0]+alpha[1])
        return builder.value(label+"_moment_residual", (minus+plus-target[0], plus-minus-target[1]), at=alpha.point)
    def step(builder, state):
        alpha, target = state["alpha"], state["target"]
        minus, plus = pm.exp(alpha[0]-alpha[1]), pm.exp(alpha[0]+alpha[1])
        n, m = minus+plus, plus-minus
        r0, r1 = n-target[0], m-target[1]; determinant = n*n-m*m
        updated = builder.value(label+"_Newton", (alpha[0]-(n*r0-m*r1)/determinant,
            alpha[1]-(-m*r0+n*r1)/determinant), at=alpha.point)
        return builder.product(alpha=updated, target=target)
    def stop(builder,state):
        defect=residual(builder,state)
        return builder.dot_all(defect,defect)<1.e-26
    result = p.iterate(carried, step, stop, max_steps=20)
    return p.require_success(result)["state"]["alpha"]


def build_t05():
    """An implicit outer fixed point invokes a compiled entropy Newton loop."""
    parts = _storage("T05_nested_entropy", ("rho", "moment"), (1., .2))
    *_, p, q, _initial_values = parts
    h = ProgramScalar(p.requested_dt()); guess = p.value("outer_seed", q.n, at=q.next.point)
    old = q.n
    def step(builder, value):
        alpha = _entropy_closure(builder, value, label="inner")
        selected_moment = pm.exp(alpha[0]+alpha[1])-pm.exp(alpha[0]-alpha[1])
        return builder.value("outer_step", (old[0], old[1]-h*selected_moment), at=value.point)
    def stop(builder, value):
        residual = builder.value("outer_residual", (value[0]-old[0], value[1]-old[1]+h*value[1]), at=value.point)
        return builder.dot_all(residual,residual) < 1.e-24
    result = p.iterate(guess, step, stop, max_steps=20)
    candidate = p.require_success(result)["state"]
    alpha = _entropy_closure(p, candidate, label="final")
    p.store_history("entropy_alpha", alpha, depth=1)
    # Implicit derivative H^-1 dM at the selected relation, never differentiation
    # of the finite sequence of Newton iterations. dM=(0,1).
    derivative = p.value("implicit_tangent", (-candidate[1]/(candidate[0]**2-candidate[1]**2),
        candidate[0]/(candidate[0]**2-candidate[1]**2)), at=q.next.point)
    p.store_history("entropy_tangent", derivative, depth=1)
    def check(runtime, actual):
        raw_alpha = runtime.history_global("entropy_alpha", 1)
        raw_tangent = runtime.history_global("entropy_tangent", 1)
        if actual is None: return {}
        a = np.asarray(raw_alpha).reshape(actual.shape); tangent=np.asarray(raw_tangent).reshape(actual.shape)
        minus, plus = np.exp(a[0]-a[1]), np.exp(a[0]+a[1])
        defect = float(max(np.max(abs(minus+plus-actual[0])),np.max(abs(plus-minus-actual[1]))))
        # Independent numerical linear algebra on the selected quadrature root,
        # not a second spelling of the authored two-by-two inverse formula.
        basis=np.array([[1.,-1.],[1.,1.]])
        weights=np.stack((minus,plus),axis=-1)
        hessian=np.einsum("...k,ki,kj->...ij",weights,basis,basis)
        right=np.broadcast_to(np.array([0.,1.]),(*weights.shape[:-1],2))
        expected=np.moveaxis(np.linalg.solve(hessian,right[...,None])[...,0],-1,0)
        derivative_error = float(np.max(abs(tangent-expected)))
        return dict(passed=defect < 1.e-11 and derivative_error < 1.e-11,
                    moment_residual=defect, implicit_tangent_error=derivative_error)
    return _finish(parts, candidate, (1.,.2/1.05**2), dt=.05, end=.1, extra=check,
        profile="T05: nested native fixed-point/Newton loops, two-velocity strictly realizable exponential closure with implicit H inverse tangent; fixed quadrature, interior moments and selected unique convex root")


def build_t06():
    """Frozen and unknown arguments are separate captured SSA values."""
    parts = _storage("T06_argument_partition", ("F_ab", "F_bb"), (1.,1.))
    *_, p, q, _initial_values = parts
    from pops.time.expressions import CoefficientExpression
    h = CoefficientExpression(p.dt.to_polynomial()); seed = p.value("unknown", q.n, at=q.next.point)
    def residual(_p, unknown, old):
        return (unknown[0]-old[0]-h*old[0]*unknown[0],
                unknown[1]-old[1]-h*unknown[1]**2)
    candidate = p.solve(LocalResidual(residual, seed, captures={"old":q.n}),
        solver=LocalNewton(tolerance=1.e-13,max_iterations=20)).consume(action=FailRun())
    x=y=1.; dt=.05
    for _ in range(2):
        x=x/(1-dt*x); y=(1-reference_math.sqrt(1-4*dt*y))/(2*dt)
    return _finish(parts,candidate,(x,y),dt=dt,end=2*dt,
        profile="T06: two off-diagonal relations F(a,b)=a*b and F(a,b)=b^2, same diagonal but separate old capture and unknown; native LocalResidual on the small positive branch")


def build_t07(*, solve_error=0.):
    """Variable-step BDF2 of u'=-u with real retained state and increment rings."""
    solve_error=float(solve_error)
    if not reference_math.isfinite(solve_error) or abs(solve_error)>.01:
        raise ValueError("T07 solve_error must be finite with magnitude at most .01")
    parts = _storage("T07_variable_BDF2", ("u", "previous_h"), (1.,0.))
    *_, p, q, _initial_values = parts
    duration = p.requested_dt()
    h = ProgramScalar(duration)
    window = p.interval(duration=duration)
    # Store-before-read is the existing native history-ring convention. The
    # initial read is CopyCurrent, but the startup branch never treats it as BDF2.
    p.store_history("bdf_state", q.n, depth=1)
    prior_h = p.value("prior_h", (q.n[1],q.n[1]))
    def bdf(builder):
        retained_increment=builder.history("bdf_increment",lag=1,space=q.n.space,
            block=q.n.block,state_ref=q.n.state_ref)
        previous_window = builder.history_interval(retained_increment)
        retained_state=builder.history("bdf_state",lag=1,space=q.n.space,
            block=q.n.block,state_ref=q.n.state_ref)
        state_window=builder.history_interval(retained_state)
        previous=state_window.bind(builder,retained_state,name="previous_state")
        previous=builder.guard("same_history_begin",previous,
            state_window.begin==previous_window.begin,action=FailRun())
        previous=builder.guard("same_history_duration",previous,
            state_window.duration==previous_window.duration,action=FailRun())
        h_previous=ProgramScalar(previous_window.duration)
        ratio = h/h_previous; alpha=(1+2*ratio)/(1+ratio); beta=ratio*ratio/(1+ratio)
        # A nonzero perturbation is an intentionally inexact solve. Its original
        # residual must enter the accepted increment once, including in history.
        solution = builder.value("BDF2_relation", (((1+ratio)*q.n[0]-beta*previous[0])/(alpha+h)+solve_error,h), at=q.next.point)
        current_rate = builder.value("current_rate", (-solution[0],0.), at=q.next.point)
        defect = builder.value("original_BDF2_defect", (alpha*solution[0]-(1+ratio)*q.n[0]+beta*previous[0]+h*solution[0],0.), at=q.next.point)
        def increment(name, value, interval, measure="increment", kind="production"):
            c = Contribution(name, value, interval, measure, kind)
            return Increment((c,), (Delivery(c, q.n, ((1,0),(0,1))),))
        ledger = bdf2_increment(builder,
            increment("current_rate", current_rate, window, measure="rate"),
            increment("previous_accepted", retained_increment, previous_window),
            h=h, h_previous=h_previous, interval=window,
            solve_residual=increment("original_defect", defect, window, kind="defect"))
        updated = ledger.apply(builder, q.n, name="delivered_BDF2_equation", at=q.next.point)
        builder.record_scalar("bdf2.history.begin", previous_window.begin)
        builder.record_scalar("bdf2.history.duration", previous_window.duration)
        return builder.value("BDF2_candidate",(updated[0],h),at=q.next.point)
    def startup(builder):
        return builder.value("BE_startup", (q.n[0]/(1+h),h), at=q.next.point)
    candidate = p.branch(p.max(prior_h)>0,bdf,startup,name="startup_or_BDF2")
    accepted_increment = p.value("full_accepted_increment", (candidate[0]-q.n[0],0.), at=q.next.point)
    p.store_history("bdf_increment", accepted_increment, depth=1)
    # FixedDt is deliberately clipped at .25: h=.1,.1,.05 gives a nonunit ratio.
    dt=.1; value=1.; old=1.; hp=0.
    for step in (.1,.1,.05):
        if hp == 0: new=value/(1+step)
        else:
            ratio=step/hp; alpha=(1+2*ratio)/(1+ratio); beta=ratio*ratio/(1+ratio)
            new=((1+ratio)*value-beta*old)/(alpha+step)+solve_error
        old,value,hp=value,new,step
    def history_check(runtime, actual):
        state = runtime.history_global("bdf_state",1)
        increment = runtime.history_global("bdf_increment",1)
        if actual is None: return {}
        state=np.asarray(state).reshape(actual.shape);increment=np.asarray(increment).reshape(actual.shape)
        error=float(np.max(abs(actual[0]-state[0]-increment[0])))
        return dict(passed=error < 2.e-12, accepted_increment_history_error=error)
    return _finish(parts,candidate,(value,.05),dt=dt,end=.25,extra=history_check,
        profile="T07: BE startup then variable BDF2 h=.1,.1,.05; actual state/step and full increment histories, weighted historical production and original finite-solve defect (solve perturbation="+repr(solve_error)+"); HistoryInterval authenticates actual retained publication; strict restart and retained-interval provenance have dedicated native reception tests; AMR transfer unqualified")


def build_t08(cells=(32,32), *, reject_parent=False, regrid=False):
    """Subcycled transport on a frozen hierarchy; optional regrid is a refusal witness."""
    if type(regrid) is not bool:
        raise TypeError("T08 regrid must be an explicit bool")
    from pops.amr import (AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid,
        AMRTagging, AMRTransfer, Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.numerics import DiscretizationPlan
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.time import every
    from pops.params import RuntimeParam
    frame, _ = _domain("T08_AMR_window",cells)
    model=pops.Model("T08_transport",frame=frame);U=model.state("u",components=("u",))
    flux=model.flux("transport",state=U,frame=frame,
        components={frame.axes[0]:(U[0],),frame.axes[1]:(0*U[0],)},
        waves={frame.axes[0]:(1.,),frame.axes[1]:(0.,)})
    rate=model.rate("balance",equation=pm.ddt(U)==-pm.div(flux))
    case=pops.Case("T08_subcycled_AMR");block=case.block("method",model)
    plan=DiscretizationPlan();plan.rates.add(rate,_fv(flux,U));case.numerics(plan,block=block)
    p=pops.Program("T08_parent_window");q=p.state(block[U]);dt=.005
    candidate=p.value("level_candidate",q.n+p.dt*rate(q.n),at=q.next.point)
    p.keep_history(q,depth=2);p.commit(q.next,candidate)
    def after_sync(builder):
        synchronized=builder.value("after_reflux_and_average",q.n,at=q.next.point)
        # This guard executes in the synchronization phase, after child work.
        condition=builder.min(synchronized) > (2. if reject_parent else 0.)
        checked=builder.guard("parent_after_children",synchronized,condition,action=RejectAttempt())
        builder.commit(q.next,checked)
    p.after_synchronization(after_sync);p.step_strategy(FixedDt(dt));case.program(p)
    subject=block[U];transfer=AMRTransfer();transfer.state(subject,StateTransfer())
    threshold=case.param(RuntimeParam("refine_threshold",default=1.15))
    layout=AMR(grid=CartesianGrid(frame=frame,cells=cells,periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2,ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(pm.ValueExpr(subject)>case.value(threshold)),Buffer(cells=1)),
            hysteresis=Hysteresis(0,EqualityPolicy.HOLD),conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=(AMRRegrid(schedule=every(1,clock=p.clock)) if regrid else AMRRegrid.frozen()),
        transfer=transfer,execution=AMRExecution.subcycled((AMRClockRelation(0,1,2),)))
    x,y=np.meshgrid((np.arange(cells[0])+.5)/cells[0],(np.arange(cells[1])+.5)/cells[1])
    initial=(1-.1*np.cos(2*np.pi*x)-.1*np.cos(2*np.pi*y))[None]
    def check(runtime):
        from .method_amr_reception import composite_base_projection, translation_metrics
        integral=runtime.integral("method",component=0)
        levels=runtime.n_levels()
        report=runtime.program_report()
        # Every rank participates in every gather before the owner evaluates
        # the composite; dense holes and covered coarse cells are never counted.
        states=[runtime.block_level_state_global("method",level) for level in range(levels)]
        projected=composite_base_projection(states,runtime.patch_boxes(),cells=cells)
        if projected is None:return non_owner_metrics()
        motion=translation_metrics(projected["means"],cells=cells,time=runtime.time())
        mass=abs(integral-1.)
        measure_error=abs(float(projected["means"].mean())-integral)
        counts=projected["active_counts"]
        interface=bool(counts[0] and any(counts[1:]))
        motion.update(passed=bool(motion["passed"] and mass<2.e-11 and measure_error<2.e-11
                                 and bool(report.histories) and levels>=2 and interface),metrics_owner=True,
            composite_mass_defect=mass,composite_measure_error=measure_error,
            levels=levels,retained_histories=len(report.histories),
            composite_active_cells=counts,
            has_active_coarse_fine_interface=interface)
        return motion
    return CaseSpec(case,layout,_initial(case,subject,initial),{"method":subject},2*dt,dt,check,
        "T08: two-level periodic Cartesian AMR advection with a central fine region and active coarse-fine interface, 2:1 recursive subcycling, native reflux/average-down and retained histories; "
        +("runtime regrid requested as an explicit unsupported deep-history remap witness; " if regrid else "initial hierarchy frozen during evolution; ")
        +"composite translation assessed independently; parent rejection injected after synchronization in separate reception")


def build_t09(**kwargs):
    """Spatial characteristic reconstruction with a native requested interval."""
    from .method_interval_transport import build_interval_transport
    return build_interval_transport(**kwargs)


def build_t09_homogeneous(*, dt=.2):
    """Frozen homogeneous BGK interval equation and one shared finite edge."""
    tau=.4;equilibrium=1.;rates=[]
    def physics(model,U):
        source=model.source("collision",on=U,value=((equilibrium-U[0])/tau,(equilibrium-U[1])/tau))
        edge=model.source("directed_edge",on=U,value=(-.3*(U[0]-U[1]),.3*(U[0]-U[1])))
        rates.append(model.rate("collision_and_edge",equation=pm.ddt(U)==edge+source))
    parts=_storage("T09_BGK_interval",("left","right"),(2.,.5),configure=physics)
    case,layout,model,block,U,p,q,initial=parts
    rate=rates[0]
    from pops.numerics import DiscretizationPlan, StateStorage
    plan=DiscretizationPlan();plan.rates.add(rate,StateStorage());case.numerics(plan,block=block)
    face=Use(rate.occurrences[0],"finite_edge","whole_interval","inventory","face")
    volume=Use(rate.occurrences[1],"cell","whole_interval","inventory","collision")
    duration=p.requested_dt()
    interval=p.interval(duration=duration)
    def body(builder,old,h):
        delta=ProgramScalar(h)
        factor=-pm.expm1(-delta/tau)
        # The equilibrium difference vanishes here. This exact homogeneous
        # characteristic interval still depends nonlinearly on the requested h.
        transfer=.3*tau*factor*(old[0]-old[1])
        return {"face":(transfer,0.),"collision":(factor*(equilibrium-old[0]),factor*(equilibrium-old[1]))}
    realized=Realize("BGK_joint_interval",owns=(face,volume),reads=(volume,),body=body,
        bindings={"old":q.n,"h":duration},outputs={face:("face",),volume:("collision",)}).evaluate(p)
    shared=Contribution("time_integrated_edge",realized[face],interval)
    collision=Contribution("time_integrated_collision",realized[volume],interval,kind="production")
    total=Increment((shared,collision),(Delivery(shared,q.n,((-1,0),(1,0))),Delivery(collision,q.n,((1,0),(0,1)))))
    candidate=total.apply(p,q.n,name="BGK_deliveries",at=q.next.point)
    factor=-reference_math.expm1(-dt/tau);j=.3*tau*factor*1.5
    expected=(2-factor-j,.5+.5*factor+j)
    return _finish(parts,candidate,expected,dt=dt,
        profile="T09: frozen homogeneous BGK Duhamel interval plus one co-located finite graph edge, jointly authored owned face and collision uses; interval dependence re-evaluated for each request, full spatial UGKS unqualified")


def build_t10(*, retry=False):
    parts=_storage("T10_relaxed_Heun",("x","y"),(1.,0.))
    *_,p,q,initial=parts
    h=p.requested_dt()
    first=p.value("rotation_first",(-q.n[1],q.n[0]))
    predictor=p.value("predictor",q.n+p.dt*first,at=q.next.point)
    second=p.value("rotation_second",(-predictor[1],predictor[0]),at=q.next.point)
    increment=p.value("Heun_increment",.5*p.dt*first+.5*p.dt*second,at=q.next.point)
    gamma=-2*p.dot_all(q.n,increment)/p.dot_all(increment,increment)
    # State and reached clock consume the identical native gamma value.
    candidate=p.value("relaxed",q.n+increment*gamma,at=q.next.point)
    if retry:
        candidate=p.guard("retry_changed_requested_interval",candidate,h<.75,action=RejectAttempt())
    p.reached_duration(gamma*h)
    requested=.5 if retry else 1.;g=1/(1+requested*requested/4)
    expected=(1-g*requested*requested/2,g*requested)
    return _finish(parts,candidate,expected,dt=1.,end=g*requested,
        strategy=ComputedDt(1.,endpoint_ulps=4,max_rejections=1 if retry else 0),
        profile="T10: native collective energy relaxation of Heun rotation, shared gamma in state and computed time frontier; proposal 1 reaches 4/5, optional rejected request 1 retries with 1/2; no spatial exchange support")


def build_t11(*, circulation=0., initial=(1.,2.)):
    """Directed processes remain separate even when their net ODE cancels."""
    if circulation < 0:raise ValueError("circulation must be nonnegative")
    if len(initial)!=2 or any(value<0 for value in initial):raise ValueError("inventories must be nonnegative")
    initial_pair=tuple(initial);total=sum(initial_pair)
    parts=_storage("T11_Patankar",("donor","receiver"),initial_pair)
    *_,p,q,initial_values=parts
    h=ProgramScalar(p.requested_dt());x,y=q.n[0],q.n[1]
    # P12=a*x+c*x*y and P21=b*y+c*x*y. The circulation cancels
    # from the ODE, but its Patankar weights use different donor inventories.
    # These are the continuous donor-rate extensions at zero inventory;
    # the simplified equations avoid an artificial 0/0 at an empty species.
    forward=.3+circulation*y
    backward=.2+circulation*x
    determinant=1+h*(forward+backward)
    candidate=p.value("Patankar_solve",(((1+h*backward)*x+h*backward*y)/determinant,
        (h*forward*x+(1+h*forward)*y)/determinant),at=q.next.point)
    x,y=initial_pair;dt=.4
    for _ in range(2):
        a=.3+circulation*y;b=.2+circulation*x;d=1+dt*(a+b)
        x,y=((1+dt*b)*x+dt*b*y)/d,(dt*a*x+(1+dt*a)*y)/d
    def check(runtime,actual):
        if actual is None:return {}
        defect=float(np.max(abs(actual[0]+actual[1]-total)))
        return dict(passed=defect<3.e-13 and np.min(actual)>=0,
                    conservation_defect=defect,minimum=float(np.min(actual)))
    return _finish(parts,candidate,(x,y),dt=dt,end=2*dt,extra=check,
        profile="T11: two-species modified Patankar-Euler from explicit directed processes; nonnegative inventories and exact shared transfer, optional cancelling circulation distinguishes equal net ODE presentations")


def build_t12(**kwargs):
    """Euler mesh-face provisional/admissibility/replacement reception."""
    from .method_euler_correction import build_euler_correction
    return build_euler_correction(**kwargs)


def build_t12_finite_edge():
    """Replace one shared total interval value, rebuilding every recipient."""
    rates=[]
    def physics(model,U):
        source=model.source("edge_presentation",on=U,value=(-.25*(U[0]-U[1]),.25*(U[0]-U[1])))
        rates.append(model.rate("balance",equation=pm.ddt(U)==source))
    parts=_storage("T12_shared_replacement",("left","right"),(2.,1.),configure=physics)
    case,layout,model,block,U,p,q,initial=parts
    rate=rates[0]
    from pops.numerics import DiscretizationPlan, StateStorage
    plan=DiscretizationPlan();plan.rates.add(rate,StateStorage());case.numerics(plan,block=block)
    use=Use(rate.occurrences[0],"finite_edge","whole_interval","inventory")
    interval=p.interval()
    def realize(name,coefficient):
        return Realize(name,owns=(use,),body=lambda builder,old,h:{"total":(
            coefficient*ProgramScalar(h)*(old[0]-old[1]),0.)},
            bindings={"old":q.n,"h":interval.duration},outputs={use:("total",)}).evaluate(p)[use]
    old=Contribution("old_total",realize("deliberately_overestimated_total",.5),interval)
    ledger=Increment((old,),(Delivery(old,q.n,((-1,0),(1,0))),))
    provisional=ledger.apply(p,q.n,name="provisional",at=q.next.point)
    p.store_history("discarded_candidate",provisional,depth=1)
    replacement=Contribution("corrected_total",realize("corrected_consistent_total",.25),old.interval)
    corrected=ledger.replace(old,replacement)
    candidate=corrected.apply(p,q.n,name="recomputed_both_recipients",at=q.next.point)
    corrected.require_application(candidate)
    dt=.2;difference=(1-2*.25*dt)**2
    expected=((3+difference)/2,(3-difference)/2)
    def check(runtime,actual):
        discarded=runtime.history_global("discarded_candidate",1)
        if actual is None:return {}
        discarded=np.asarray(discarded).reshape(actual.shape)
        defect=float(np.max(abs(actual.sum(axis=0)-3.)))
        change=float(np.max(abs(actual-discarded)))
        return dict(passed=defect<3.e-13 and change>1.e-4,
                    conservation_defect=defect,replacement_changes_candidate=change)
    return _finish(parts,candidate,expected,dt=dt,end=2*dt,extra=check,
        profile="T12: one shared finite edge per cell, occurrence-qualified total interval realization, replace rather than add, both incidence deliveries rebuilt and version checked; Cartesian/AMR face-ledger replacement remains unqualified")


def build_method(method, *, profile=None, require_complete=False, **kwargs):
    from .profile_dispatch import build_profile
    key = str(method).upper()
    if key not in READY:
        raise ValueError("unknown method " + str(method))
    return build_profile(key, profile=profile, require_complete=require_complete, **kwargs)
