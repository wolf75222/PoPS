"""Global nonsymmetric Schur, IMEX and selected nonlinear relations in Python.

All algorithms record ordinary PoPS maps, reductions, products and native loops.
Operator callbacks are elaborated into that graph; no callback executes during
a native step. Linear acceptance reconstructs the original residual, including
after nested elimination. The examples use periodic cell arrays of caller size.
"""
from __future__ import annotations

import math
import pops
from pops import math as pm
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Product, type_of

from .common import CaseSpec
from .models_M01_M14 import _domain
from .scientific_krylov_v14 import gmres


def amap(p, a, body):
    return p.map(p.domain(a), body)


def axpy(p, a, factor, b):
    return amap(p, a, lambda c, i: c.at(a, i)+factor*c.at(b, i))


def dot(p, a, b):
    values = amap(p, a, lambda c, i: c.at(a, i)*c.at(b, i))
    return p.reduce(values, p.float64(0), lambda c, total, x: total+x, order="ascending")


def at(p, a, j):
    return p.at(a, p.index(p.domain(a), j))


def periodic_operator(p, u, *, diagonal, diffusion, advection, length=1.):
    """diagonal*I - diffusion*Delta_h + advection*D_backward."""
    n = p.size(p.domain(u))
    scale = p.float64(n)/length
    def cell(b, i):
        j = b.index_value(i)
        center = b.at(u, i)
        left, right = at(b, u, (j+n-1)%n), at(b, u, (j+1)%n)
        return diagonal*center+diffusion*scale*scale*(2*center-left-right)+advection*scale*(center-left)
    return amap(p, u, cell)


def solve_linear(p, rhs, action, *, capacity, tolerance=2.e-12):
    zero = amap(p, rhs, lambda b, i: b.float64(0))
    answer, iterations = gmres(p, rhs, zero, action,
                              atol=p.float64(tolerance*tolerance), capacity=capacity)
    defect = axpy(p, rhs, -1, action(p, answer))
    squared = dot(p, defect, defect)
    p.require_success(p.branch(squared <= 64*tolerance*tolerance,
        lambda b: b.success(squared), lambda b: b.failure(squared, 20401)))
    return answer, squared, iterations


def solve_blocks(p, f, g, *, A, B, C, D, method, capacity, tolerance=2.e-12):
    """Solve [A B; C D] [x;y]=[f;g] by either algorithm, with original checks."""
    if method == "schur":
        @graph(inputs=Product(rhs=type_of(g)), output=type_of(g))
        def inverse_graph(b, args):
            return solve_linear(b, args["rhs"], D, capacity=capacity,
                                tolerance=tolerance/20)[0]
        def inverse(b, value):
            return inverse_graph(b, rhs=value)
        rhs = axpy(p, f, -1, B(p, inverse(p, g)))
        def reduced(b, x):
            return axpy(b, A(b, x), -1, B(b, inverse(b, C(b, x))))
        x = solve_linear(p, rhs, reduced, capacity=capacity, tolerance=tolerance)[0]
        y = inverse(p, axpy(p, g, -1, C(p, x)))
    elif method == "monolithic":
        n = p.size(p.domain(f))
        pair = p.range_set(2*n, capacity=2*capacity, ownership="replicated")
        def pack(b, x, y):
            return b.map(pair, lambda c, i: c.branch(c.index_value(i) < n,
                lambda d: at(d, x, c.index_value(i)),
                lambda d: at(d, y, c.index_value(i)-n)))
        def unpack(b, values):
            return (amap(b, f, lambda c, i: at(c, values, c.index_value(i))),
                    amap(b, f, lambda c, i: at(c, values, n+c.index_value(i))))
        def action(b, values):
            x, y = unpack(b, values)
            return pack(b, axpy(b, A(b, x), 1, B(b, y)), axpy(b, C(b, x), 1, D(b, y)))
        values = solve_linear(p, pack(p, f, g), action, capacity=2*capacity,
                              tolerance=tolerance)[0]
        x, y = unpack(p, values)
    else:
        raise ValueError("method must be 'schur' or 'monolithic'")
    r0 = axpy(p, axpy(p, A(p, x), 1, B(p, y)), -1, f)
    r1 = axpy(p, axpy(p, C(p, x), 1, D(p, y)), -1, g)
    squared = dot(p, r0, r0)+dot(p, r1, r1)
    p.require_success(p.branch(squared < 256*tolerance*tolerance,
        lambda b: b.success(squared), lambda b: b.failure(squared, 20402)))
    return x, y, squared


def _context(name, count, dt, steps, initialize, step):
    if type(count) is not int or count < 3 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("positive duration and at least three periodic cells required")
    if type(steps) is not int or steps < 1:
        raise ValueError("positive integer step count required")
    _, layout = _domain(name+"_context", (4, 4))
    p = pops.Program(name)
    domain = p.range_set(count, capacity=count, ownership="replicated")
    state = initialize(p, domain)
    initial = p.product(domain=domain, values=state, time=p.float64(0), step=p.int64(0))
    schema = ContinuationSchema(initial, fields=(), time=("time",), step=("step",))
    base = p.continuation_open(initial, schema)
    old = p.continuation_read(base)
    result, residual = step(p, old["values"], p.requested_dt())
    continued = p.product(domain=old["domain"], values=result,
                         time=old["time"]+p.requested_dt(), step=old["step"]+p.int64(1))
    envelope = p.product(continuation=continued, increment=p.product(),
                         observations=p.product(residual=residual), checks=p.product())
    proposal = p.product(h=p.requested_dt())
    candidate = p.continuation_freeze(base, proposal, envelope)
    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal), candidate=type_of(envelope)),
           output=Product(accepted=Bool, checks=Product(residual=Float64)))
    def assess(b, a):
        r = a["candidate"]["observations"]["residual"]
        return b.product(accepted=r < 1.e-18, checks=b.product(residual=r))
    p.continuation_publish(p.continuation_accept(candidate, p.continuation_attest(candidate, assess)))
    p.step_strategy(FixedDt(dt))
    case = pops.Case(name)
    case.program(p)
    return CaseSpec(case, layout, {}, {}, steps*dt, dt,
        lambda runtime: dict(passed=runtime.macro_step() == steps, frontier_only=True),
        name+": complete native array continuation; independently checked original global residual")


def build_t04_nonsymmetric(*, count=5, steps=2, dt=.1):
    def initialize(p, domain):
        x = p.map(domain, lambda b, i: 1+.2*pm.cos(2*math.pi*b.float64(b.index_value(i))/count))
        y = p.map(domain, lambda b, i: .3+.15*pm.sin(2*math.pi*b.float64(b.index_value(i))/count))
        return p.product(x=x, y=y, monolithic_x=x, monolithic_y=y)
    def step(p, old, _h):
        x, y = old["x"], old["y"]
        A = lambda b, value: periodic_operator(b, value, diagonal=2., diffusion=.015, advection=.15)
        D = lambda b, value: periodic_operator(b, value, diagonal=1.5, diffusion=.03, advection=.04)
        def coupling(b, value, first):
            return amap(b, value, lambda c, i: c.at(value, i)*(
                .2*(1+.1*pm.cos(2*math.pi*c.float64(c.index_value(i))/count)) if first
                else -.3*(1+.2*pm.sin(2*math.pi*c.float64(c.index_value(i))/count))))
        B, C = lambda b, value: coupling(b, value, True), lambda b, value: coupling(b, value, False)
        xs, ys, rs = solve_blocks(p, x, y, A=A, B=B, C=C, D=D,
                                 method="schur", capacity=count)
        xm, ym, rm = solve_blocks(p, x, y, A=A, B=B, C=C, D=D,
                                 method="monolithic", capacity=count)
        return p.product(x=xs, y=ys, monolithic_x=xm, monolithic_y=ym), rs+rm
    return _context("T04_global_nonsymmetric_nested_Schur", count, dt, steps, initialize, step)


def solve_selected_relation(p, old, forcing, *, capacity, mode, branch, tolerance=2.e-12):
    """F1(a,b)=a*b versus F2(a,b)=b*b, plus the SAME global spatial term.

    F2 admits distinct positive and negative solutions. Branch selection is an
    explicit predicate on every component. The frozen data never change during
    Newton or its inner GMRES. Failed line search/selection is a native failure.
    """
    if mode not in ("frozen", "unknown") or branch not in (-1, 1):
        raise ValueError("invalid relation partition or selected sign")
    def spatial(b, value):
        return periodic_operator(b, value, diagonal=0., diffusion=.002, advection=.01)
    def residual(b, value):
        local = amap(b, value, lambda c, i:
            (c.at(old, i) if mode == "frozen" else c.at(value, i))*c.at(value, i)-c.at(forcing, i))
        return axpy(b, local, 1, spatial(b, value))
    def selected(b, value):
        return b.reduce(value, b.boolean(True), lambda c, valid, x:
            c.branch(valid, lambda d: branch*x > 0, lambda d: d.boolean(False)), order="ascending")
    seed = amap(p, forcing, lambda b, i: branch*pm.sqrt(b.at(forcing, i)))
    def step(b, value):
        r = residual(b, value)
        norm = dot(b, r, r)
        def jacobian(c, direction):
            diagonal = amap(c, direction, lambda d, i:
                (d.at(old, i) if mode == "frozen" else 2*d.at(value, i))*d.at(direction, i))
            return axpy(c, diagonal, 1, spatial(c, direction))
        correction = solve_linear(b, r, jacobian, capacity=capacity, tolerance=tolerance/20)[0]
        initial = b.product(scale=b.float64(1), x=axpy(b, value, -1, correction))
        def reduce_step(c, trial):
            scale = trial["scale"]/2
            return c.product(scale=scale, x=axpy(c, value, -scale, correction))
        def accept(c, trial):
            rr = residual(c, trial["x"])
            return c.branch(selected(c, trial["x"]),
                lambda d: dot(d, rr, rr) <= (1-.0001*trial["scale"])*norm,
                lambda d: d.boolean(False))
        trial = b.require_success(b.iterate(initial, reduce_step, accept, max_steps=24))["state"]
        return trial["x"]
    def stop(b, value):
        r = residual(b, value)
        return dot(b, r, r) <= tolerance*tolerance
    solved = p.require_success(p.iterate(seed, step, stop, max_steps=12))["state"]
    p.require_success(p.branch(selected(p, solved), lambda b: b.success(solved),
                               lambda b: b.failure(solved, 20601)))
    r = residual(p, solved)
    return solved, dot(p, r, r)


def build_t06_selected_global(*, count=5, mode="unknown", branch=1):
    def initialize(p, domain):
        data = p.map(domain, lambda b, i: 2+.1*pm.sin(2*math.pi*b.float64(b.index_value(i))/count))
        forcing = p.map(domain, lambda b, i: 1+.2*pm.cos(2*math.pi*b.float64(b.index_value(i))/count))
        return p.product(frozen=data, forcing=forcing, root=data)
    def step(p, old, _h):
        root, r = solve_selected_relation(p, old["frozen"], old["forcing"],
                                          capacity=count, mode=mode, branch=branch)
        return p.product(frozen=old["frozen"], forcing=old["forcing"], root=root), r
    return _context("T06_global_"+mode+("_positive" if branch == 1 else "_negative"),
                    count, .1, 1, initialize, step)


def imex_ars222(p, old, h, explicit, implicit, *, capacity):
    """ARS(2,2,2) tableau, gamma=1-1/sqrt(2), delta=1-1/(2 gamma).

    Three stage rows, common abscissae (0,gamma,1), with the last row accepted.
    Primary tableau: Ascher--Ruuth--Spiteri, section 2.6, page 9,
    https://steveruuth.org/wp-content/uploads/2020/10/ars.pdf .
    ``implicit`` is a supplied linear action; its solve is globally coupled.
    The four additive order-two conditions are independently tested.
    """
    gamma = 1-1/math.sqrt(2)
    delta = 1-1/(2*gamma)
    e0 = explicit(p, old)
    def action(b, value):
        return axpy(b, value, -h*gamma, implicit(b, value))
    first, r0, _ = solve_linear(p, axpy(p, old, h*gamma, e0), action, capacity=capacity)
    rhs = axpy(p, old, h*delta, e0)
    rhs = axpy(p, rhs, h*(1-delta), explicit(p, first))
    rhs = axpy(p, rhs, h*(1-gamma), implicit(p, first))
    answer, r1, _ = solve_linear(p, rhs, action, capacity=capacity)
    return answer, r0+r1


def build_t02_imex_second_order(*, count=7, dt=.025, steps=4):
    def initialize(p, domain):
        return p.map(domain, lambda b, i: 1+.2*pm.cos(2*math.pi*b.float64(b.index_value(i))/count))
    def step(p, old, h):
        explicit = lambda b, value: periodic_operator(b, value, diagonal=-.2, diffusion=0., advection=-.3)
        implicit = lambda b, value: periodic_operator(b, value, diagonal=0., diffusion=-.05, advection=0.)
        return imex_ars222(p, old, h, explicit, implicit, capacity=count)
    return _context("T02_IMEX_ARS222_global_diffusion", count, dt, steps, initialize, step)
