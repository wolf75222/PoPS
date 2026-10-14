"""N01 obstruction: preserving two moments requires an additional coarse value.

All coordinates, volumes, q and time below are nondimensional.  Cell averages
obey m1=<q>, m2=<q**2>, Hbar=m1+m2.  A single coarse scalar constrained by
m1=1 has square1, whereas the fine data q=(0,2), volumes=(1/2,1/2) have m2=2.
The enriched representation stores variance=m2-m1**2, so Hbar=q+q**2+variance.
The Program executes the remap and its checks; Python only authors equations.
This witness does not replace the separate BDF2 or distributed-repartition tests.
"""
from __future__ import annotations

import pops
from pops._ir.quantity import PhysicalDimension, PhysicalSupport
from pops.bundles import BundleSchema, Remap, remap
from pops.continuations import Advance, Assess, ContinuationSchema, advance_once
from pops.graph_functions import graph
from pops.model.spaces import Space
from pops.relations import Check
from pops.representations import ScientificType, ScientificValue
from pops.time import FixedDt
from pops.time.variant_types import Bool, Failure, Float64, Product, Result, type_of
from .common import CaseSpec
from .models_M01_M14 import _domain


def _sum(p, values):
    return p.reduce(values, p.float64(0), lambda g, a, b: a+b, order='ascending')


def _second(p, bundle, i, enriched):
    q = p.at(bundle['fields']['q'], i)
    return q*q + (p.at(bundle['fields']['variance'], i) if enriched else 0)


def _integral(p, bundle, body):
    return _sum(p, p.map(bundle['instance'], lambda g, i:
        g.at(bundle['geometry']['volume'], i)*body(g, i)))


def _check_moments(p, before, after, enriched):
    def mean(g, bundle):
        return _integral(g, bundle, lambda b, i: b.at(bundle['fields']['q'], i))
    def second(g, bundle):
        return _integral(g, bundle, lambda b, i: _second(b, bundle, i, enriched))
    old_mean, old_second = mean(p, before), second(p, before)
    new_mean, new_second = mean(p, after), second(p, after)
    errors = dict(mean=new_mean-old_mean, second=new_second-old_second,
        accumulation=new_mean+new_second-old_mean-old_second,
        volume=_sum(p, after['geometry']['volume'])-_sum(p, before['geometry']['volume']))
    for name in ('mean', 'second', 'accumulation'):
        target = _integral(p, before,
            lambda g, i, name=name: g.at(before['histories'][name], i))
        errors['history_'+name] = _integral(p, after,
            lambda g, i, name=name: g.at(after['histories'][name], i))-target
    accepted = p.boolean(True)
    for value in errors.values():
        accepted = p.branch(value*value < 1.e-26,
            lambda g, accepted=accepted: accepted, lambda g: g.boolean(False))
    return Check(p, accepted=accepted, observations=p.product(**errors))


def build_coarsening_moments_v14(*, mode='enriched', steps=3):
    """Warm up one fine step, then attempt a complete conservative coarsening."""
    if mode not in ('mean_only', 'enriched'):
        raise ValueError('choose mean_only or enriched')
    if type(steps) is not int or steps < 2:
        raise ValueError('at least two steps expose the coarsening obligation')
    enriched = mode == 'enriched'
    _, layout = _domain('coarsening_moments_context', (4, 4))
    case = pops.Case('coarsening_moments_'+mode)
    p = pops.Program('coarsening_original_moment_equations')
    domain = p.range_set(2, capacity=2, ownership='replicated')
    q = p.map(domain, lambda g, i: 2*g.float64(g.index_value(i)))
    zero = p.map(domain, lambda g, i: g.float64(0))
    one = p.map(domain, lambda g, i: g.float64(1))
    fields = {'q': q, **({'variance': zero} if enriched else {})}
    geometry = p.product(volume=p.map(domain, lambda g, i: g.float64(.5)),
        center=p.map(domain, lambda g, i: (g.float64(g.index_value(i))+.5)/2))
    indices = p.map(domain, lambda g, i: i)
    maps = p.product(source=domain, left=indices, right=indices, left_weight=one, right_weight=zero)
    # Independent retained sample (1,3), not reconstructed from current q=(0,2).
    history_q = p.map(domain, lambda g, i: g.at(q, i)+1)
    histories = p.product(mean=history_q,
        second=p.map(domain, lambda g, i: g.at(history_q, i)**2),
        accumulation=p.map(domain, lambda g, i: g.at(history_q, i)+g.at(history_q, i)**2))
    raw = p.product(instance=domain, fields=p.product(**fields), geometry=geometry,
                    maps=maps, histories=histories)
    bindings = {('fields', name): 'current' for name in fields}
    bindings.update({('geometry', name): 'current' for name in ('volume', 'center')})
    bindings.update({('histories', name): 'current' for name in ('mean', 'second', 'accumulation')})
    bindings.update({('maps', name): 'current' for name in ('left', 'right', 'left_weight', 'right_weight')})
    bindings.update({('maps', name, '[]'): 'source' for name in ('left', 'right')})
    bundle_schema = BundleSchema(raw, domains={'current': ('instance',), 'source': ('maps', 'source')},
                                 bindings=bindings)
    bundle = bundle_schema.bind(p, raw)
    initial = p.product(bundle=bundle, time=p.float64(0), step=p.int64(0))
    space = Space('nondimensional_finite_cell_scalar', ('value',),
        support=PhysicalSupport((('x', 'unit_interval'),)), units=(PhysicalDimension(),),
        value_shape=(), sampling='cell_average', representation='coordinates')
    def scientific(value, *, array=True):
        return ScientificType(type_of(value), space, 'finite_cell_scalar', (), (PhysicalDimension(),),
            {'name': 'explicit_finite_volume_weights'}, dofs=type_of(domain) if array else None)
    declarations = {('time',): scientific(initial['time'], array=False)}
    for role, names in (('fields', tuple(fields)), ('geometry', ('volume', 'center')),
                        ('maps', ('left_weight', 'right_weight')),
                        ('histories', ('mean', 'second', 'accumulation'))):
        for name in names:
            declarations[('bundle', role, name)] = scientific(bundle[role][name])
    schema = ContinuationSchema(initial, fields=(), time=('time',), step=('step',),
                                scientific=declarations)
    base = p.continuation_open(initial, schema)
    requested_h = p.requested_dt()
    proposal = p.product(h=ScientificValue(requested_h, scientific(requested_h, array=False)).value)
    defects = Product(**{name: Float64 for name in ('mean', 'second', 'accumulation', 'volume',
        'history_mean', 'history_second', 'history_accumulation')})
    errors = Failure(moment_loss=defects)
    remap_result = Result(type_of(bundle), errors)

    @graph(inputs=Product(bundle=type_of(bundle), data=Product(coarsen=Bool)), output=remap_result)
    def transfer(g, a):
        old = a['bundle']; source = old['instance']
        coarsen = g.branch(a['data']['coarsen'], lambda b: b.size(source) == 2,
                          lambda b: b.boolean(False))
        count = g.branch(coarsen, lambda b: b.int64(1), lambda b: b.size(source))
        target = g.range_set(count, family=source)
        left = g.map(target, lambda b, i: b.index(source, b.branch(coarsen,
            lambda c: c.int64(0), lambda c: c.index_value(i))))
        right = g.map(target, lambda b, i: b.index(source, b.branch(coarsen,
            lambda c: c.int64(1), lambda c: c.index_value(i))))
        wl = g.map(target, lambda b, i: b.branch(coarsen, lambda c: c.float64(.5), lambda c: c.float64(1)))
        wr = g.map(target, lambda b, i: 1-b.at(wl, i))
        def weighted(b, i, equation):
            return b.at(wl, i)*equation(b, b.at(left, i))+b.at(wr, i)*equation(b, b.at(right, i))
        mean = g.map(target, lambda b, i: weighted(b, i, lambda c, j: c.at(old['fields']['q'], j)))
        second = g.map(target, lambda b, i: weighted(b, i, lambda c, j: _second(c, old, j, enriched)))
        after_fields = {'q': mean}
        if enriched:
            after_fields['variance'] = g.map(target, lambda b, i: b.at(second, i)-b.at(mean, i)**2)
        volumes = g.map(target, lambda b, i: b.branch(coarsen,
            lambda c: c.at(old['geometry']['volume'], c.at(left, i))+c.at(old['geometry']['volume'], c.at(right, i)),
            lambda c: c.at(old['geometry']['volume'], c.at(left, i))))
        centers = g.map(target, lambda b, i: weighted(b, i, lambda c, j: c.at(old['geometry']['center'], j)))
        retained = {name: g.map(target, lambda b, i, name=name:
            weighted(b, i, lambda c, j: c.at(old['histories'][name], j)))
            for name in ('mean', 'second', 'accumulation')}
        after = bundle_schema.bind(g, g.product(instance=target, fields=g.product(**after_fields),
            geometry=g.product(volume=volumes, center=centers),
            maps=g.product(source=source, left=left, right=right, left_weight=wl, right_weight=wr),
            histories=g.product(**retained)))
        checked = _check_moments(g, old, after, enriched)
        return g.branch(checked['accepted'], lambda b: b.variant(remap_result, 'ok', after),
            lambda b: b.variant(remap_result, 'err', b.variant(errors, 'moment_loss', checked['observations'])))

    method = Remap(transfer, source_instance=('maps', 'source'),
                   histories={name: 'transfer' for name in ('mean', 'second', 'accumulation')})
    envelope = Product(continuation=schema.write_type, increment=Product(), observations=defects, checks=Product())
    outcome = Result(envelope, errors)

    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal)), output=outcome)
    def trial(g, a):
        before = a['base']
        after = g.require_success(remap(g, method, before['bundle'], data=g.product(coarsen=before['step'] >= 1)))
        checked = _check_moments(g, before['bundle'], after, enriched)
        state = g.product(bundle=after, time=before['time']+a['proposal']['h'], step=before['step']+1)
        return g.variant(outcome, 'ok', g.product(continuation=state, increment=g.product(),
            observations=checked['observations'], checks=g.product()))

    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal), candidate=envelope),
           output=Product(accepted=Bool, checks=defects))
    def assessment(g, a):
        before = a['base']; after = a['candidate']['continuation']
        checked = _check_moments(g, before['bundle'], after['bundle'], enriched)
        accepted = g.branch(after['time'] == before['time']+a['proposal']['h'],
            lambda b: b.branch(after['step'] == before['step']+1,
                lambda c: checked['accepted'], lambda c: c.boolean(False)), lambda b: b.boolean(False))
        return g.product(accepted=accepted, checks=checked['observations'])

    accepted = p.require_success(advance_once(p, Advance(trial, Assess(assessment)), base=base, proposal=proposal))
    p.continuation_publish(accepted)
    p.step_strategy(FixedDt(.1)); case.program(p)
    return CaseSpec(case, layout, {}, {}, .1*steps, .1,
        lambda runtime: {'passed': runtime.macro_step() == steps, 'frontier_only': True},
        'N01 nondimensional moments: scalar coarsening obstruction versus explicit variance enrichment')
