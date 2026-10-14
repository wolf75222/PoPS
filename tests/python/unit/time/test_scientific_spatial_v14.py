"""Source integration of N03. Native scientific reception has a separate test file."""
import pops
import pytest
from examples.api_v02.scientific_spatial_v14 import build_scientific_spatial_v14


@pytest.mark.parametrize("krylov", ("cg", "gmres"))
def test_full_spatial_newton_cg_graph_is_scientifically_checked_and_block_free(krylov):
    spec = build_scientific_spatial_v14(steps=1, krylov=krylov)
    program = spec.case._time
    assert not program._block_indices() and not spec.initial_values
    roots = [v for v in program._values if v.op == 'continuation_open']
    assert len(roots) == 1 and not roots[0].attrs['schema']['fields']
    annotated = [v for v in program._values if v.attrs.get('scientific_graph_v14', {}).get('mode') == 'strict']
    assert annotated and annotated[-1].vtype == 'array'
    # This is the closed finite method body, not an opaque name or output-only annotation.
    methods = [str(v.attrs['scientific_graph_v14']['graph']) for v in annotated]
    assert any(all(op in body for op in ('structured_iterate','finite_at','capture_result'))
               for body in methods)
    assert any(v.op == 'continuation_publish' for v in program._values)
    pops.validate(spec.case)


def test_branch_can_capture_an_array_from_a_lexical_ancestor():
    p = pops.Program("lexical_array")
    domain = p.range_set(3, capacity=3, ownership="replicated")
    def body(b):
        outer = b.map(domain, lambda c, i:c.float64(2))
        return b.map(domain, lambda c, i:c.branch(c.index_value(i)==0,
            lambda d:outer, lambda d:outer))
    result = p.require_success(p.capture(body))
    assert result.vtype == 'array'


def test_branch_cannot_import_array_from_a_sibling_region():
    p = pops.Program("sibling_array")
    domain = p.range_set(3, capacity=3, ownership="replicated")
    escaped = []
    def first(b):
        value = b.map(domain, lambda c, i:c.float64(2))
        escaped.append(value)
        return value
    allowed = p.branch(p.boolean(True), first, lambda b:b.map(domain, lambda c,i:c.float64(0)))
    with pytest.raises(ValueError, match='region|escape|sub-block|branch'):
        p.branch(p.boolean(True), lambda b:escaped[0], lambda b:allowed)


def _contribution_packet(mode):
    from pops._ir.quantity import PhysicalDimension, PhysicalSupport
    from pops.frames import Cartesian1D
    from pops.math import ddt
    from pops.model.spaces import Space
    from pops.graph_functions import graph
    from pops.representations import ScientificType, ScientificValue, ScientificMap
    from pops.continuations import ContinuationSchema
    from pops.realization import Use, Realize, require_continuation_coverage
    from pops.time.variant_types import Bool, Float64, Product, type_of
    p = pops.Program('exact_finite_delivery')
    d = p.range_set(2, capacity=2, ownership='replicated')
    q = p.map(d, lambda b, i:b.float64(1))
    space = Space('dimensionless', ('q',), support=PhysicalSupport(()),
                  units=(PhysicalDimension(),), value_shape=())
    scalar = ScientificType(Float64, space, 'point_value', (), (), {'name':'counting'})
    array = ScientificType(type_of(q), space, 'point_value', (), (), {'name':'counting'}, type_of(d))
    initial = p.product(q=q, time=p.float64(0), step=p.int64(0))
    schema = ContinuationSchema(initial, fields=(), time=('time',), step=('step',),
                               scientific={('q',):array, ('time',):scalar})
    base = p.continuation_open(initial, schema); current = p.continuation_read(base)
    h = ScientificValue(p.float64(.1), scalar)
    old = ScientificValue(current['q'], array)
    body = graph(inputs=Product(q=array.finite_type, h=Float64), output=array.finite_type)(
        lambda b,a:b.map(b.domain(a['q']), lambda c,i:a['h']*c.at(a['q'], i)))
    mapping = ScientificMap(body, inputs={'q':array,'h':scalar}, output=array, strict=True)
    model = pops.Model('growth_equations', frame=Cartesian1D())
    state = model.state('u', components=('u',))
    source = model.source('growth', on=state, value=(state[0],))
    rate = model.rate('growth_balance', equation=ddt(state)==source)
    use = Use(rate.occurrences[0], 'cells', 'interval_increment', 'point_value', scientific_type=array)
    value = Realize('growth_step', owns=(use,), bindings={'q':old,'h':h}, outputs={use:()},
        body=lambda b,q,h:mapping(b,q=q,h=h), strict=True).evaluate(p)[use]
    new = p.map(d, lambda b,i:b.at(old.value, i)+b.at(value, i))
    if mode == 'diagnostic_only': new = old.value
    if mode == 'branch_omits':
        new = p.branch(current['step'] == 0, lambda b:new, lambda b:old.value)
    packet = p.product(q=new, time=current['time']+h.value, step=current['step']+1)
    proposal = p.product(h=h.value)
    envelope = p.product(continuation=packet, increment=p.product(),
                         observations=p.product(diagnostic=value), checks=p.product())
    candidate = p.continuation_freeze(base, proposal, envelope)
    assess = graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal), candidate=type_of(envelope)),
        output=Product(accepted=Bool, checks=Product()))(lambda b,a:b.product(accepted=b.boolean(True),checks=b.product()))
    publication = p.continuation_publish(p.continuation_accept(candidate, p.continuation_attest(candidate, assess)))
    require_continuation_coverage(publication, {('missing' if mode == 'wrong_path' else 'q',):(use,)})
    from pops.realization import validate_realized_outputs
    validate_realized_outputs(p)
    return p


def test_exact_finite_contribution_is_delivered_to_scientific_packet():
    _contribution_packet('valid')


@pytest.mark.parametrize('mode', ('diagnostic_only','wrong_path','branch_omits'))
def test_diagnostics_or_missing_branch_do_not_deliver_finite_contribution(mode):
    with pytest.raises((ValueError,TypeError), match='path|deliver|coverage|contribution|missing'):
        _contribution_packet(mode)
