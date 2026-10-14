"""Native semantic checks for the algebra, independent of the physical-model corpus."""
from __future__ import annotations

import numpy as np
import pops
from pops.algebra import sin, cos, log, expm1, where, jvp, vjp, Pairings
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt

from .common import CaseSpec, state_array, non_owner_metrics


def build_algebra_check(kind="elementary"):
    frame = Rectangle("algebra", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("algebra_semantics", frame=frame)
    state = model.state("samples", components=("a", "b", "c"))
    case = pops.Case("algebra_" + kind)
    block = case.block("samples", model)
    numerics = DiscretizationPlan()
    numerics.states.add(state, StateStorage())
    case.numerics(numerics, block=block)
    P = pops.Program("algebra_" + kind)
    q = P.state(block[state])
    initial = np.broadcast_to(np.linspace(-.7, .7, 8)[None, None, :], (3, 4, 8)).copy()
    initial[1] += .2
    initial[2] += .1
    if kind == "elementary":
        a, b, c = (q.n[i] for i in range(3))
        result = P.value("elementary", (sin(a) + cos(b), expm1(c),
            where(a > 0, lambda: log(a), lambda: expm1(a))), at=q.next.point)
        expected = initial.copy()
        expected[0] = np.sin(initial[0]) + np.cos(initial[1])
        expected[1] = np.expm1(initial[2])
        expected[2] = np.expm1(initial[0])
        positive = initial[0] > 0
        expected[2][positive] = np.log(initial[0][positive])
    elif kind in ("jvp", "vjp"):
        initial += 1.5
        a,b,c=(q.n[i] for i in range(3))
        outputs=(a*a*sin(b)+c,log(a)+expm1(b),a**b+c*c)
        x,y,z=initial
        jacobian=np.array([[2*x*np.sin(y),x*x*np.cos(y),np.ones_like(x)],
                           [1/x,np.exp(y),np.zeros_like(x)],
                           [y*x**(y-1),x**y*np.log(x),2*z]])
        if kind=="jvp":
            directions=(.3,-.2,.1)
            expressions=jvp(outputs,wrt=(a,b,c),direction=directions)
            expected=np.einsum("ijyx,j->iyx",jacobian,np.array(directions))
        else:
            dual=(.4,.7,-.2)
            pairings=Pairings(output_to_covector=lambda v:tuple(w*x for w,x in zip((7,11,13),v)),
                              input_from_covector=lambda v:tuple(x/w for w,x in zip((2,3,5),v)))
            expressions=vjp(outputs,wrt=(a,b,c),covector=dual,pairings=pairings)
            expected=np.einsum("ijyx,i->jyx",jacobian,np.array(dual)*np.array((7,11,13)))/np.array((2,3,5))[:,None,None]
        result=P.value("differential",expressions,at=q.next.point)
    elif kind == "simultaneous_product":
        other = P.value("other", tuple(q.n[i] + 10 for i in range(3)))
        pair = P.product(left=q.n, right=other, count=P.int64(0))
        def swap(p, carried, index):
            return p.product(left=carried["right"], right=carried["left"],
                             count=carried["count"] + 1)
        folded = P.fold(pair, 3, swap)
        result = P.value("swapped", folded["left"], at=q.next.point)
        expected = initial + 10
    elif kind in ("nested_iteration", "budget_fallback", "budget_reject"):
        carry = P.product(q=q.n, count=P.int64(0))
        def step(p, carried):
            def add(pp, value, index):
                return pp.value("inner_add", tuple(value[i] + i + 1 for i in range(3)))
            advanced = p.fold(carried["q"], 2, add)
            return p.product(q=advanced, count=carried["count"] + 1)
        iteration = P.iterate(carry, step, lambda p, s: s["count"] >= 3,
                              max_steps=3 if kind == "nested_iteration" else 2)
        if kind == "budget_fallback":
            value = P.match(iteration,
                ok=lambda p, ok: ok["state"]["q"],
                err=lambda p, err: p.value("fallback", tuple(q.n[i] + 100 for i in range(3))))
            expected = initial + 100
        else:
            value = P.require_success(iteration)["state"]["q"]
            expected = initial + np.array([6, 12, 18])[:, None, None]
        result = P.value("iteration_result", value, at=q.next.point)
    else:
        raise ValueError("unknown algebra reception " + kind)
    P.commit(q.next, result)
    P.step_strategy(FixedDt(.01))
    case.program(P)
    subject = block[state]
    case.initials.add(InitialCondition(state=subject, value=BindArray(),
                                       projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 4), periodic=PeriodicAxes(frame.axes)))
    def invariants(runtime):
        values = state_array(runtime, "samples", initial.shape)
        if values is None:
            return non_owner_metrics()
        error = float(np.max(np.abs(values - expected)))
        return {"passed": bool(np.all(np.isfinite(values)) and error < 4e-13),
                "max_error": error, "metrics_owner": True}
    return CaseSpec(case, layout, {subject: initial}, {"samples": subject}, .01, .01,
                    invariants, "native algebra semantic check: " + kind)
