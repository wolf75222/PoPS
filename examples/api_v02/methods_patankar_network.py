"""Directed N-species production/destruction and MPRK22(1), wholly authored.

Kopecz--Meister, https://arxiv.org/html/1702.04589v1, definition (19),
Theorem 3.5: the predictor uses pi=y_n and the final denominator sigma=y_stage.
Rates remain directed process data, including pairs cancelling in the net ODE.
The two positive stage weights are both 1/2. No rate or matrix is evaluated by
Python during a native step; only this author's graph construction runs there.
"""
from __future__ import annotations

import math
import pops
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Product, type_of

from .common import CaseSpec
from .models_M01_M14 import _domain
from .methods_coupled_nonsymmetric import amap, at, dot, axpy, solve_linear


def directed_processes(p, inventory, *, capacity, quotient):
    """One process per ordered pair; quotient(p, y, donor, receiver) >= 0.

    The author supplies the continuous donor-rate extension at empty inventory.
    Self-processes cancel between the identical endpoint contributions. Rates
    are never inferred from a net source. The full process array is native data.
    """
    n = p.size(p.domain(inventory))
    edges = p.range_set(n*n, capacity=capacity*capacity, ownership="replicated")
    def edge(b, i):
        k = b.index_value(i)
        donor, receiver = k//n, k%n
        rate = quotient(b, inventory, donor, receiver)
        rate = b.require_success(b.branch(rate >= 0,
            lambda c: c.success(rate), lambda c: c.failure(rate, 21101)))
        return b.product(donor=donor, receiver=receiver, quotient=rate,
                         production=rate*at(b, inventory, donor))
    return p.map(edges, edge)


def _net(p, inventory, processes, values):
    def balance(b, i):
        target = b.index_value(i)
        def term(c, total, edge):
            amount = edge["quotient"]*at(c, values, edge["donor"])
            incoming = c.branch(edge["receiver"] == target, lambda d: amount, lambda d: d.float64(0))
            outgoing = c.branch(edge["donor"] == target, lambda d: amount, lambda d: d.float64(0))
            return total+incoming-outgoing
        return b.reduce(processes, b.float64(0), term, order="ascending")
    return amap(p, inventory, balance)


def modified_patankar22(p, inventory, h, *, capacity, quotient):
    """Conservative two-stage second-order MPRK for caller-defined processes.

    The predictor solves with donor-rate extensions, hence permits zero initial
    species. The corrector requires a positive stage denominator for each
    nonzero weighted production. An identically zero process contributes zero;
    a positive production with zero denominator fails explicitly. No clipping,
    epsilon denominator or alternative method is substituted.
    """
    first = directed_processes(p, inventory, capacity=capacity, quotient=quotient)
    stage, _, _ = solve_linear(p, inventory,
        lambda b, value: axpy(b, value, -h, _net(b, inventory, first, value)), capacity=capacity)
    second = directed_processes(p, stage, capacity=capacity, quotient=quotient)
    # Each call issued its own domain; do not address the second with an index
    # from the first merely because both cardinalities happen to be equal.
    def average(b, i):
        left = b.at(first, i)
        right = at(b, second, b.index_value(i))
        denominator = at(b, stage, left["donor"])
        production = .5*(left["production"]+right["production"])
        def zero(c):
            return c.require_success(c.branch(production == 0,
                lambda d: d.success(d.float64(0)), lambda d: d.failure(d.float64(0), 21102)))
        rate = b.branch(denominator > 0, lambda c: production/denominator, zero)
        return b.product(donor=left["donor"], receiver=left["receiver"], quotient=rate,
                         production=production)
    averaged = p.map(p.domain(first), average)
    answer, residual, _ = solve_linear(p, inventory,
        lambda b, value: axpy(b, value, -h, _net(b, inventory, averaged, value)), capacity=capacity)
    transfers = p.map(p.domain(averaged), lambda b, i: b.product(
        donor=b.at(averaged, i)["donor"], receiver=b.at(averaged, i)["receiver"],
        amount=h*b.at(averaged, i)["quotient"]*at(b, answer, b.at(averaged, i)["donor"])))
    # The same per-process amount is delivered once to each endpoint. Returning
    # it permits a separate ledger check against the accepted inventory delta.
    return answer, transfers, residual


def network_quotient(circulation=0.):
    """A nonuniform reversible ring plus explicitly cancelling nonlinear pairs."""
    if not math.isfinite(circulation) or circulation < 0:
        raise ValueError("nonnegative finite circulation required")
    def quotient(p, values, donor, receiver):
        n = p.size(p.domain(values))
        forward = p.branch(receiver == (donor+1)%n,
            lambda b: .3+.03*b.float64(donor), lambda b: b.float64(0))
        backward = p.branch(receiver == (donor+n-1)%n,
            lambda b: .15+.02*b.float64(receiver), lambda b: b.float64(0))
        pair = p.branch(receiver != donor,
            lambda b: circulation*at(b, values, receiver), lambda b: b.float64(0))
        return forward+backward+pair
    return quotient


def build_t11_network(*, count=5, dt=.1, steps=6, circulation=0., empty=False):
    """Three simultaneously advanced refinements receive genuine temporal order.

    Each accepted macro interval advances independent coarse/medium/fine values
    using 1/2/4 native MPRK steps respectively. A single GraphFunction contains
    the algorithm. All three states reach exactly the same physical time.
    """
    if type(count) is not int or count < 3 or type(steps) is not int or steps < 1:
        raise ValueError("at least three species and positive step count required")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("positive finite duration required")
    quotient = network_quotient(circulation)
    _, layout = _domain("Patankar_network_context", (4, 4))
    p = pops.Program("MPRK22_general_directed_network")
    domain = p.range_set(count, capacity=count, ownership="replicated")
    def initialize(b, i):
        value = 1+.2*b.float64(b.index_value(i))
        return b.branch(b.index_value(i) == 0,
            lambda c: c.float64(0) if empty else value, lambda c: value)
    values = p.map(domain, initialize)
    # Stage/edge domains remain inside the step and need no fabricated identity
    # in the persistent schema. Only the species arrays and clock are retained.
    initial = p.product(domain=domain, coarse=values, medium=values, fine=values,
                        time=p.float64(0), step=p.int64(0), residual=p.float64(0))
    schema = ContinuationSchema(initial, fields=(), time=("time",), step=("step",))
    base = p.continuation_open(initial, schema)
    old = p.continuation_read(base)
    array = type_of(values)
    @graph(inputs=Product(values=array, h=Float64), output=Product(values=array, residual=Float64))
    def method(b, args):
        answer, transfers, residual = modified_patankar22(b, args["values"], args["h"],
                                                         capacity=count, quotient=quotient)
        def balance(c, i):
            j = c.index_value(i)
            net = c.reduce(transfers, c.float64(0), lambda d, total, edge:
                total+d.branch(edge["receiver"] == j, lambda e: edge["amount"], lambda e: e.float64(0))
                -d.branch(edge["donor"] == j, lambda e: edge["amount"], lambda e: e.float64(0)), order="ascending")
            return c.at(answer, i)-c.at(args["values"], i)-net
        defect = amap(b, answer, balance)
        residual = residual+dot(b, defect, defect)
        return b.product(values=answer, residual=residual)
    outcomes = {}
    h = p.requested_dt()
    for name, substeps in (("coarse", 1), ("medium", 2), ("fine", 4)):
        seed = p.product(values=old[name], residual=p.float64(0))
        outcomes[name] = p.fold(seed, substeps,
            lambda b, state, _k, substeps=substeps: method(b, values=state["values"], h=h/substeps),
            max_steps=substeps)
    residual = sum(outcome["residual"] for outcome in outcomes.values())
    continued = p.product(domain=old["domain"],
        **{name: outcome["values"] for name, outcome in outcomes.items()},
        time=old["time"]+h, step=old["step"]+p.int64(1), residual=residual)
    envelope = p.product(continuation=continued, increment=p.product(), observations=p.product(), checks=p.product())
    proposal = p.product(h=h)
    candidate = p.continuation_freeze(base, proposal, envelope)
    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal), candidate=type_of(envelope)),
           output=Product(accepted=Bool, checks=Product(residual=Float64)))
    def assess(b, a):
        current = a["candidate"]["continuation"]
        valid = b.boolean(True)
        for name in ("coarse", "medium", "fine"):
            valid = b.reduce(current[name], valid, lambda c, ok, x:
                c.branch(ok, lambda d: x >= 0, lambda d: d.boolean(False)), order="ascending")
            before = b.reduce(a["base"][name], b.float64(0), lambda c, total, x: total+x, order="ascending")
            after = b.reduce(current[name], b.float64(0), lambda c, total, x: total+x, order="ascending")
            valid = b.branch(valid, lambda c: (after-before)**2 < 1.e-20, lambda c: c.boolean(False))
        valid = b.branch(valid, lambda c: current["residual"] < 1.e-20, lambda c: c.boolean(False))
        return b.product(accepted=valid, checks=b.product(residual=current["residual"]))
    p.continuation_publish(p.continuation_accept(candidate, p.continuation_attest(candidate, assess)))
    p.step_strategy(FixedDt(dt))
    case = pops.Case("T11_N_species_MPRK22"); case.program(p)
    return CaseSpec(case, layout, {}, {}, steps*dt, dt,
        lambda runtime: dict(passed=runtime.macro_step() == steps, frontier_only=True),
        "T11: N-species directed MPRK22(1), shared edge transfers, 1/2/4 substep native "
        "refinements, positivity and conservation acceptance; donor rate extensions explicit")
