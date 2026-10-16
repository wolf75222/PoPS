"""Path-conservative transport inside native controls, followed by a composite solve.

The two independent fields deliberately require different AMR transfers: exact
injection for the first-order path method, and conservative linear transfer for
the diffusion discretization. The composition does not pretend these two
spatial discretizations have an interchangeable transfer contract.
"""
from fractions import Fraction

import numpy as np
import pops
from pops import math as pm
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import AMR
from pops.lib.amr import StateTransfer, ConservativeInjection, CoarseFineInjection
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import (Diffusion, DiscretizationPlan, PathConservativeFiniteVolume,
                           SymbolicPath, reconstruction, riemann, variables)
from pops.params import RuntimeParam
from pops.time import AdaptiveCFL, FailRun, ImplicitStage, DerivativeStrategy, SolveUnknown
from pops.solvers import Newton
from .common import CaseSpec, non_owner_metrics
from .models_M01_M14 import _initial


def build_path_hierarchy_composition(*, mode="branch", implicit=True, cells=8, dt=1.e-4, levels=2):
    frame = Rectangle("composition_box", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("authored_transport_diffusion", frame=frame)
    state = model.state("U", components=("u", "v"))
    u, v = state
    flux = model.flux("transport", state=state, frame=frame,
                      components={x: tuple(.5*q for q in state), y: (0., 0.)})
    product = model.nonconservative_product("nonconservative_action", state=state,
        matrices={x: ((v*v, 0.), (0., 0.)), y: ((0., 0.), (0., 0.))},
        conservative_components=("v",))
    path = SymbolicPath(product, frame=frame,
        quadrature=((0., 1./6.), (.5, 4./6.), (1., 1./6.)),
        speed=lambda left, right, axis:
            .5+pm.maximum(left[1]*left[1], right[1]*right[1]) if axis == 0 else 0.)
    diffusive_model = pops.Model("authored_diffusion", frame=frame)
    diffusive_state = diffusive_model.state("Z", components=("z",))
    diffusion = diffusive_model.diffusive_flux("constitutive_flux", state=diffusive_state,
        value=.02*pm.grad(diffusive_state[0]))
    path_rate = model.rate("balance", equation=pm.ddt(state) == -pm.div(flux)-product)
    diffusion_rate = diffusive_model.rate("balance", equation=pm.ddt(diffusive_state) == pm.div(diffusion))
    plan = DiscretizationPlan()
    plan.rates.add(path_rate, PathConservativeFiniteVolume(
        flux=flux, path=path, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    diffusion_plan = DiscretizationPlan()
    diffusion_plan.rates.add(diffusion_rate, Diffusion(flux=diffusion))
    case = pops.Case("path_continuation_composition")
    block = case.block("transport", model)
    instance = block[state]
    case.numerics(plan, block=block)
    diffusion_block = case.block("diffusion", diffusive_model)
    diffusion_instance = diffusion_block[diffusive_state]
    case.numerics(diffusion_plan, block=diffusion_block)
    p = pops.Program("authored_path_composition")
    q = p.state(instance)
    z = p.state(diffusion_instance)
    initial = p.value("initial_iterate", 1*q.n, at=q.next.point)
    count = 2 if mode in {"fold", "unrolled"} else 1

    def update(b, current):
        rhs = b.value("path_rhs", path_rate(current), at=current.point)
        predictor = b.value("explicit_predictor", current+(Fraction(1, count)*p.dt)*rhs,
                            at=current.point)
        return predictor

    if mode == "fold":
        result = p.fold(initial, count, lambda b, carried, index: update(b, carried))
    elif mode == "branch":
        result = p.branch(p.max(initial) > .5, lambda b: update(b, initial), lambda b: initial)
    elif mode == "unrolled":
        result = update(p, update(p, initial))
    elif mode == "straight":
        result = update(p, initial)
    else:
        raise ValueError("unknown composition mode")
    p.commit(q.next, result)
    if implicit:
        request = ImplicitStage(diffusion_rate, z.n, p.dt).request(
            unknown=SolveUnknown("state", z.n), seed=z.n,
            derivative=DerivativeStrategy("finite_difference"))
        solved = p.solve(request, solver=Newton(tolerance=1.e-12, max_iterations=12,
            linear_tolerance=1.e-10, linear_max_iterations=120, restart=30)).consume(action=FailRun())[0]
        p.commit(z.next, p.value("final_diffusion", 1*solved, at=z.next.point))
    else:
        p.commit(z.next, p.value("unchanged_diffusion", 1*z.n, at=z.next.point))
    p.step_strategy(AdaptiveCFL(cfl=.25, max_dt=dt))
    case.program(p)
    transfer = AMRTransfer()
    transfer.state(instance, StateTransfer(prolongation=ConservativeInjection(), coarse_fine=CoarseFineInjection()))
    transfer.state(diffusion_instance, StateTransfer())
    threshold = case.param(RuntimeParam("refinement_threshold", default=.3))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=(cells, cells),
                                   periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=levels, ratios=(2,)*(levels-1)),
        tagging=AMRTagging(rules=(Tag(pm.ValueExpr(instance)["v"] > case.value(threshold)), Buffer(cells=0)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid.frozen(), transfer=transfer, execution=AMRExecution.synchronous())
    xx, yy = np.meshgrid((np.arange(cells)+.5)/cells, (np.arange(cells)+.5)/cells)
    data = np.stack((1+.1*np.sin(2*np.pi*xx)+.05*np.cos(2*np.pi*yy),
                     .3+.1*np.cos(2*np.pi*xx)))
    initial_values = _initial(case, instance, data)
    initial_values.update(_initial(case, diffusion_instance, data[:1].copy()))

    def check(runtime):
        values = runtime.block_level_state_global("transport", 0)
        diffusive_values = runtime.block_level_state_global("diffusion", 0)
        if values is None:
            return non_owner_metrics()
        actual = np.asarray(values).reshape(data.shape)
        diffusive = np.asarray(diffusive_values).reshape(data[:1].shape)
        finite = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(diffusive)))
        # v has no nonconservative row. This test's periodic diffusion also
        # conserves its scalar mass; no conservation claim is made for u.
        error = max(abs(actual[1].mean()-data[1].mean()),
                    abs(diffusive.mean()-data[0].mean()))
        return dict(passed=finite and error < 5.e-11, metrics_owner=True,
                    conservation_error=float(error))

    return CaseSpec(case, layout, initial_values,
        {"transport": instance, "diffusion": diffusion_instance}, dt, dt, check,
        "Python path transport under %s, then composite implicit diffusion" % mode)
