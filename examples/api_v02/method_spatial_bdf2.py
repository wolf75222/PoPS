"""BDF2 heat evolution, authored as one global implicit relation per step.

The startup is theta=2/3 (local error O(h²)), followed by fixed-step BDF2.
Both relations share tau=2h/3; only their explicitly authored previous member
differs. No nonlinear solve is hidden inside a branch or replaced by Python.
"""
from fractions import Fraction
import numpy as np
import pops
from pops import math as pm
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging,
    AMRTransfer, Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.layouts import AMR, Uniform
from pops.lib.amr import StateTransfer
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan
from pops.params import RuntimeParam
from pops.solvers import Newton
from pops.time import DerivativeStrategy, FailRun, FixedDt, ImplicitStage, SolveUnknown
from .common import CaseSpec, non_owner_metrics
from .models_M01_M14 import _domain, _initial


def build_spatial_bdf2(cells=(12, 12), *, dt=.01, steps=8, hierarchy="uniform",
                       history_control=False):
    if hierarchy not in {"uniform", "full", "partial"}:
        raise ValueError("hierarchy must be uniform, full, or partial")
    if dt <= 0 or type(steps) is not int or steps < 2:
        raise ValueError("BDF2 reception requires positive dt and at least two steps")
    frame, _ = _domain("spatial_bdf2_domain", cells)
    model = pops.Model("diffusive_balance", frame=frame)
    U = model.state("u", components=("u",))
    flux = model.diffusive_flux("constitutive_flux", state=U, value=.1*pm.grad(U[0]))
    rate = model.rate("balance", equation=pm.ddt(U) == pm.div(flux))
    case = pops.Case("global_spatial_bdf2")
    block = case.block("method", model)
    plan = DiscretizationPlan(); plan.rates.add(rate, Diffusion(flux=flux))
    case.numerics(plan, block=block)
    p = pops.Program("authored_spatial_BDF2"); q = p.state(block[U])
    p.keep_history(q, depth=1)
    window = p.interval()
    def multistep(builder):
        retained = q.prev.value
        interval = builder.history_interval(retained)
        previous = interval.bind(builder, retained, name="previous_accepted_state")
        # The negative control deliberately consumes the current state twice.
        if history_control: previous = q.n
        return builder.value("BDF2_previous_member",
            Fraction(4,3)*q.n-Fraction(1,3)*previous, at=q.next.point)
    def startup(builder):
        return builder.value("consistent_theta_startup",
            q.n+Fraction(1,3)*p.dt*rate(q.n), at=q.next.point)
    previous = p.branch(window.begin > 0, multistep, startup, name="startup_or_BDF2")
    coordinates = p.value("spatial_coordinates", q.n, at=q.next.point)
    request = ImplicitStage(rate, previous, Fraction(2,3)*p.dt).request(
        unknown=SolveUnknown("state", coordinates), seed=q.n,
        derivative=DerivativeStrategy("finite_difference"))
    solved = p.solve(request, solver=Newton(tolerance=1.e-12, max_iterations=20,
        linear_tolerance=1.e-10, linear_max_iterations=200, restart=40)).consume(action=FailRun())[0]
    p.commit(q.next, p.value("accepted_BDF2", solved, at=q.next.point))
    p.step_strategy(FixedDt(float(dt))); case.program(p)
    grid = CartesianGrid(frame=frame, cells=cells, periodic=PeriodicAxes(frame.axes))
    layout = Uniform(grid)
    if hierarchy != "uniform":
        transfer = AMRTransfer(); transfer.state(block[U], StateTransfer())
        threshold = case.param(RuntimeParam("refinement_threshold",
            default=.5 if hierarchy == "full" else 1.18))
        layout = AMR(grid=grid, hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(rules=(Tag(pm.ValueExpr(block[U]) > case.value(threshold)), Buffer(cells=1)),
                hysteresis=Hysteresis(0,EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
            regrid=AMRRegrid.frozen(), transfer=transfer, execution=AMRExecution.synchronous())
    x,y = np.meshgrid((np.arange(cells[0])+.5)/cells[0], (np.arange(cells[1])+.5)/cells[1])
    initial = 1+.3*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
    values = _initial(case, block[U], initial[None])
    def check(runtime):
        mass = runtime.integral("method", component=0)
        if mass is None: return non_owner_metrics()
        error = abs(float(mass)-1.)
        return dict(passed=error < 3.e-11, metrics_owner=True, mass_error=error)
    return CaseSpec(case, layout, values, {"method":block[U]}, steps*dt, dt, check,
        "fixed-step BDF2 with consistent theta=2/3 startup; actual composite spatial Newton; "
        + hierarchy + " Cartesian periodic diffusion, retained accepted State")
