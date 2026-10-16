"""Native dependent finite state: conservative refinement and scalar decay.

Every live value, geometry entry, parent map and retained history is held by the
PoPS Continuation.  The Uniform layout supplies an execution context only; this
Case declares no model, block, State, MultiFab carrier or Python stepping loop.
This is a finite hierarchy witness, not a claim about legacy AMR sidecars.
"""
from __future__ import annotations

import pops
from pops.bundles import BundleSchema, Remap, remap
from pops.continuations import ContinuationSchema, Advance, Assess, advance_once
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Int64, Product, Result, Failure, type_of
from .common import CaseSpec
from .models_M01_M14 import _domain


def _sum(p, values):
    return p.reduce(values, p.float64(0), lambda g, a, b: a+b, order='ascending')


def _mass(p, bundle, member='q', *, history=False):
    values = bundle['histories' if history else 'fields'][member]
    return _sum(p, p.map(bundle['instance'], lambda g, i:
        g.at(values, i)*g.at(bundle['geometry']['volume'], i)))


def build_bundle_remap_v14(*, mode='success', steps=3, capacity=64):
    """Refine each finite cell into two; transfer the whole old-state history.

    q'=-q uses explicit Euler; refinement copies the parent density and halves
    its volume.  The independent receiver checks every entry, including maps.
    Deliberate stale-history, capacity and parent-rejection cases must roll back.
    """
    if mode not in {'success', 'stale_history', 'parent_reject', 'capacity'}:
        raise ValueError('unknown Bundle receiver mode')
    if type(steps) is not int or steps < 1 or type(capacity) is not int or capacity < 2:
        raise ValueError('positive step count and capacity >= 2 are required')
    _, layout = _domain('bundle_native_context', (4, 4))
    case = pops.Case('native_bundle_remap_v14_'+mode)
    p = pops.Program('bundle_refinement_equations')
    domain = p.range_set(2, capacity=capacity, ownership='replicated')
    q = p.map(domain, lambda g, i: g.float64(1)+g.float64(g.index_value(i)))
    volume = p.map(domain, lambda g, i: g.float64(.5))
    center = p.map(domain, lambda g, i: (g.float64(g.index_value(i))+.5)/2)
    parent = p.map(domain, lambda g, i: i)
    raw = p.product(instance=domain, fields=p.product(q=q),
        geometry=p.product(volume=volume, center=center),
        maps=p.product(source=domain, parent=parent), histories=p.product(previous=q))
    bundle_schema = BundleSchema(raw,
        domains={'current': ('instance',), 'source': ('maps', 'source')},
        bindings={('fields', 'q'): 'current', ('geometry', 'volume'): 'current',
            ('geometry', 'center'): 'current', ('maps', 'parent'): 'current',
            ('maps', 'parent', '[]'): 'source', ('histories', 'previous'): 'current'})
    bundle = bundle_schema.bind(p, raw)
    initial = p.product(bundle=bundle, time=p.float64(0), step=p.int64(0))
    schema = ContinuationSchema(initial, fields=(), time=('time',), step=('step',))
    base = p.continuation_open(initial, schema)
    proposal = p.product(h=p.requested_dt())
    remap_result = Result(type_of(bundle), Failure())

    @graph(inputs=Product(bundle=type_of(bundle), data=type_of(proposal)), output=remap_result)
    def refine(g, args):
        old = args['bundle']; old_domain = old['instance']; h = args['data']['h']
        count = (g.branch(g.size(old_domain) > 2, lambda b: b.int64(capacity+1),
                         lambda b: 2*b.size(old_domain)) if mode == 'capacity' else 2*g.size(old_domain))
        new_domain = g.range_set(count, family=old_domain)
        indices = g.map(new_domain, lambda b, i: b.index(old_domain, b.index_value(i)//2))
        transferred = g.map(new_domain, lambda b, i: b.at(old['fields']['q'], b.at(indices, i)))
        history = (g.branch(g.size(old_domain) > 2, lambda b: old['fields']['q'],
                           lambda b: transferred) if mode == 'stale_history' else transferred)
        values = g.map(new_domain, lambda b, i:
            (1-h)*b.at(old['fields']['q'], b.at(indices, i)))
        volumes = g.map(new_domain, lambda b, i:
            b.at(old['geometry']['volume'], b.at(indices, i))/2)
        centers = g.map(new_domain, lambda b, i:
            (b.float64(b.index_value(i))+.5)/b.float64(b.size(new_domain)))
        result = bundle_schema.bind(g, g.product(instance=new_domain,
            fields=g.product(q=values), geometry=g.product(volume=volumes, center=centers),
            maps=g.product(source=old_domain, parent=indices), histories=g.product(previous=history)))
        return g.variant(remap_result, 'ok', result)

    remapping = Remap(refine, source_instance=('maps', 'source'), histories={'previous': 'transfer'})
    envelope = Product(continuation=schema.write_type,
        increment=Product(mass_change=Float64), observations=Product(old_mass=Float64), checks=Product())
    trial_result = Result(envelope, Failure())

    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal)), output=trial_result)
    def trial(g, args):
        before = args['base']; h = args['proposal']['h']
        after = g.require_success(remap(g, remapping, before['bundle'], data=args['proposal']))
        old_mass = _mass(g, before['bundle']); new_mass = _mass(g, after)
        continuation = g.product(bundle=after, time=before['time']+h, step=before['step']+g.int64(1))
        return g.variant(trial_result, 'ok', g.product(continuation=continuation,
            increment=g.product(mass_change=new_mass-old_mass),
            observations=g.product(old_mass=old_mass), checks=g.product()))

    checked = Product(accepted=Bool, checks=Product(balance=Float64, history=Float64, volume=Float64))

    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal), candidate=envelope), output=checked)
    def assessment(g, args):
        before = args['base']; after = args['candidate']['continuation']; h = args['proposal']['h']
        balance = _mass(g, after['bundle'])-(1-h)*_mass(g, before['bundle'])
        history = _mass(g, after['bundle'], 'previous', history=True)-_mass(g, before['bundle'])
        volume = _sum(g, after['bundle']['geometry']['volume'])-1
        tests = (balance*balance < 1.e-26, history*history < 1.e-26, volume*volume < 1.e-26,
                 after['time'] == before['time']+h,
                 g.size(after['bundle']['instance']) == 2*g.size(before['bundle']['instance']))
        accepted = g.boolean(True)
        for check in tests:  # static composition of five native predicates
            accepted = g.branch(check, lambda b, accepted=accepted: accepted, lambda b: b.boolean(False))
        return g.product(accepted=accepted, checks=g.product(balance=balance, history=history, volume=volume))

    approved = p.require_success(advance_once(p, Advance(trial, Assess(assessment)), base=base, proposal=proposal))
    if mode == 'parent_reject':
        # The complete remapped child has been accepted provisionally, but cannot publish.
        errors = Failure(parent=Product(code=Int64))
        refused = Result(type_of(approved), errors)
        p.require_success(p.branch(p.continuation_read(base)['step'] > 0,
            lambda b: b.variant(refused, 'err', b.variant(errors, 'parent', b.product(code=b.int64(77)))),
            lambda b: b.variant(refused, 'ok', approved)))
    p.continuation_publish(approved)
    p.step_strategy(FixedDt(.1)); case.program(p)
    def check(runtime):
        # Full structural/numerical assertions live in the native receiver, not in an
        # always-true scientific certificate supplied to the Program.
        return {'passed': runtime.macro_step() == steps and abs(runtime.time()-.1*steps) < 1.e-14,
                'frontier_only': True}
    return CaseSpec(case, layout, {}, {}, .1*steps, .1, check,
        'Native zero-block finite Bundle: Euler decay, conservative binary refinement, complete history/map/geometry')
