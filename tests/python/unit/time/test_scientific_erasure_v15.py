"""Mandatory scientific checks on the actual Program validation route."""
import pytest
import pops
import pops.math as math
from pops.continuations import Candidate, ContinuationSchema
from pops.graph_functions import graph
from pops.representations import ScientificMap, ScientificValue
from pops.scientific_graph import validate_scientific_program
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Product, type_of
from tests.python.unit.time.test_scientific_graph_v14 import typ, L, T


def complete(body):
    p = pops.Program('scientific_declared_dependency')
    initial = p.product(q=p.float64(2), time=p.float64(0), step=p.int64(0))
    schema = ContinuationSchema(initial, fields=(), time=('time',), step=('step',))
    base = p.continuation_open(initial, schema)
    old = p.continuation_read(base)
    value = body(p, old['q'])
    proposal = p.product(h=p.float64(1))
    after = p.product(q=value, time=old['time']+1, step=old['step']+1)
    candidate = Candidate(p, base=base, proposal=proposal, continuation=after,
        increment=p.product(), observations=p.product(), checks=p.product())
    envelope = type_of(p.continuation_candidate_value(candidate))

    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal), candidate=envelope),
           output=Product(accepted=Bool, checks=Product()))
    def policy(g, args):
        return g.product(accepted=g.boolean(True), checks=g.product())

    p.continuation_publish(p.continuation_accept(candidate, p.continuation_attest(candidate, policy)))
    p.step_strategy(FixedDt(1))
    return p


@pytest.mark.parametrize('body,match', (
    (lambda p, q: math.exp(ScientificValue(q, typ(L)).value), 'dimensionless'),
    (lambda p, q: ScientificValue(q, typ(L)).value + ScientificValue(p.float64(3), typ(T)).value,
     'dimensions'),
    (lambda p, q: p.branch(p.boolean(True),
        lambda b: ScientificValue(q, typ(L)).value,
        lambda b: ScientificValue(b.float64(3), typ(T)).value), 'dimensions'),
))
def test_root_validation_rejects_erased_derived_annotations(body, match):
    with pytest.raises(TypeError, match=match):
        complete(body).validate()


def test_valid_scientific_dependencies_and_raw_finite_programs_remain_legal():
    for body in (lambda p, q: ScientificValue(q, typ(L)).value * 2,
                 lambda p, q: math.exp(q)):
        complete(body).validate()


def legacy_map(input_type, output_type, body):
    fn = graph(inputs=Product(q=Float64), output=Float64)(body)
    return ScientificMap(fn, inputs={'q': input_type}, output=output_type, strict=False)


def test_explicit_legacy_boundary_remains_unchecked_but_typed_downstream():
    # The admitted legacy body is deliberately dimensionally wrong. This is
    # explicitly labelled, and must never count as a normative proof.
    legacy = legacy_map(typ(L), typ(L), lambda p, a: math.exp(a['q']))
    p = pops.Program('legacy_admission')
    out = legacy(p, q=ScientificValue(p.float64(2), typ(L))).value
    assert validate_scientific_program(p)
    with pytest.raises(TypeError, match='legacy'):
        validate_scientific_program(p, strict=True)
    math.exp(out)
    with pytest.raises(TypeError, match='dimensionless'):
        validate_scientific_program(p)


def test_legacy_admission_cannot_hide_an_unrelated_invalid_expression():
    legacy = legacy_map(typ(L), typ(L), lambda p, a: 2*a['q'])
    p = pops.Program('legacy_not_global_exemption')
    q = ScientificValue(p.float64(2), typ(L))
    math.exp(q.value)
    legacy(p, q=q)
    with pytest.raises(TypeError, match='dimensionless'):
        validate_scientific_program(p)


def test_legacy_boundary_rejects_changed_numerical_body():
    legacy = legacy_map(typ(L), typ(L), lambda p, a: 2*a['q'])
    p = pops.Program('legacy_body_identity')
    out = legacy(p, q=ScientificValue(p.float64(2), typ(L))).value
    p._replace_value(out, attrs=dict(out.attrs, fn='add'))
    with pytest.raises(TypeError, match='different applied body'):
        validate_scientific_program(p)


def test_default_validation_checks_unused_partial_scientific_operations():
    p = pops.Program('partial_dead_value')
    math.exp(ScientificValue(p.float64(2), typ(L)).value)
    with pytest.raises(TypeError, match='dimensionless'):
        validate_scientific_program(p)
