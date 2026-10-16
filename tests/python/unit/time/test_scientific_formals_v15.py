"""Scientific formal/région contracts survive real graph codecs, without native execution.

Standalone durable regressions: no imports of test helpers or external overlay paths.
"""
import json
import pytest
import pops
import pops.math as math
from pops._ir.quantity import PhysicalDimension, PhysicalSupport
from pops.model.spaces import Space
from pops.continuations import Candidate, ContinuationSchema
from pops.graph_functions import GraphFunction, graph
from pops.representations import ScientificMap, ScientificType, ScientificValue
from pops.scientific_graph import validate_scientific_program
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Product, type_of
from pops.time._program.detach import detach_compiled_program

L = PhysicalDimension((("length", 1),))
T = PhysicalDimension((("time", 1),))


def typ(unit):
    space = Space("q", ("q",), support=PhysicalSupport(()), units=(unit,),
                  value_shape=(), sampling="point", representation="coordinates", domain="real")
    return ScientificType(Float64, space, "point_value", (), (), {"name":"counting"})

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


def legacy_map(input_type, output_type, body):
    fn = graph(inputs=Product(q=Float64), output=Float64)(body)
    return ScientificMap(fn, inputs={'q': input_type}, output=output_type, strict=False)


def folded(p,a,shape):
    if shape == "direct":
        return p.fold(a["q"],1,lambda g,x,i:ScientificValue(x,typ(T)).value)
    initial=p.product(q=a["q"])
    result=p.fold(initial,1,lambda g,x,i:g.product(q=ScientificValue(x["q"],typ(T)).value))
    return result["q"]


@pytest.mark.parametrize("shape",("direct","product"))
@pytest.mark.parametrize("reload",(False,True))
def test_known_region_argument_cannot_be_reannotated(shape,reload):
    f=graph(inputs=Product(q=Float64),output=Float64)(lambda p,a:folded(p,a,shape))
    with pytest.raises(TypeError,match="dimensions|annotation|scientific|Space"):
        if reload:
            f=GraphFunction.from_data(json.loads(json.dumps(f.to_data())))
        ScientificMap(f,inputs={"q":typ(L)},output=typ(L),strict=True)


@pytest.mark.parametrize("operation",("body_exp","stop_relabel"))
def test_iterate_context_cannot_disappear_on_unannotated_output(operation):
    def body(p,a):
        step=(lambda g,x:math.exp(x)) if operation=="body_exp" else (lambda g,x:x)
        stop=(lambda g,x:g.boolean(True)) if operation=="body_exp" else (lambda g,x:ScientificValue(x,typ(T)).value > 0)
        return p.require_success(p.iterate(a["q"],step,stop,max_steps=2))["state"]
    f=graph(inputs=Product(q=Float64),output=Float64)(body)
    with pytest.raises(TypeError,match="dimension|annotation|scientific|Space"):
        ScientificMap(f,inputs={"q":typ(L)},output=typ(L),strict=True)


def test_valid_known_region_identity_is_preserved():
    f=graph(inputs=Product(q=Float64),output=Float64)(
        lambda p,a:p.fold(a["q"],1,lambda g,x,i:ScientificValue(x,typ(L)).value))
    ScientificMap(f,inputs={"q":typ(L)},output=typ(L),strict=True)


def test_valid_known_region_identity_contract_survives_json():
    from pops.scientific_graph import check_scientific_contract_data
    f=graph(inputs=Product(q=Float64),output=Float64)(
        lambda p,a:p.fold(a["q"],1,lambda g,x,i:ScientificValue(x,typ(L)).value))
    mapping=ScientificMap(f,inputs={"q":typ(L)},output=typ(L),strict=True)
    check_scientific_contract_data(json.loads(json.dumps(mapping.checked_contract.to_data())))


def test_formal_input_cannot_be_redeclared_with_different_units():
    @graph(inputs=Product(q=Float64),output=Float64)
    def relabel(p,a):
        return ScientificValue(a['q'],typ(T)).value
    with pytest.raises(TypeError,match='dimension|annotation|scientific|Space'):
        ScientificMap(relabel,inputs={'q':typ(L)},output=typ(T),strict=True)


def nested():
    inner=legacy_map(typ(L),typ(L),lambda p,a: a['q']*2)
    @graph(inputs=Product(q=Float64),output=Float64)
    def outer_body(p,a):
        value=inner(p,q=ScientificValue(a['q'],typ(L)))
        return value.value/3
    # Exercise the real function codec before the second call site is emitted.
    outer_body=GraphFunction.from_data(outer_body.to_data())
    return ScientificMap(outer_body,inputs={'q':typ(L)},output=typ(L),strict=False)


def test_nested_legacy_boundaries_survive_reload_and_whole_program_detach():
    outer=nested()
    p=complete(lambda g,q:outer(g,q=ScientificValue(q,typ(L))).value)
    p.validate()
    copied=detach_compiled_program(p)
    assert copied._ir_hash()==p._ir_hash()
    copied.validate()
    with pytest.raises(TypeError,match='legacy'):
        validate_scientific_program(copied,strict=True)


def test_new_output_annotation_cannot_extend_a_legacy_scope():
    outer=nested();p=pops.Program('scope_extension')
    value=outer(p,q=ScientificValue(p.float64(3),typ(L))).value
    with pytest.raises(TypeError,match='legacy|dimensionless'):
        ScientificValue(math.exp(value),typ(L))
    with pytest.raises(TypeError,match='dimensionless'):
        validate_scientific_program(p)


def test_copied_boundary_on_new_expression_is_rejected_after_exact_reload():
    outer=nested();p=pops.Program('copied_scope')
    value=outer(p,q=ScientificValue(p.float64(3),typ(L))).value
    extra=math.exp(value)
    p._replace_value(extra,attrs=dict(extra.attrs,**{k:v for k,v in value.attrs.items()
        if k.startswith('scientific_')}))
    with pytest.raises(TypeError,match='different applied body'):
        validate_scientific_program(p)


def test_declared_length_can_be_published_alongside_an_explicit_time_proposal():
    p=complete(lambda p,q:ScientificValue(q,typ(L)).value*2)
    # requested_dt is an execution clock coordinate until its units are
    # declared by the case, just as an unannotated bound Float64 parameter.
    p.requested_dt()
    p.validate()



def test_loader_deferral_cannot_escape_to_program_or_strict_validation():
    from pops.graph_functions import _formals
    function = graph(inputs=Product(q=Float64), output=Float64)(
        lambda p, a: p.fold(a["q"], 1,
            lambda g, x, i: ScientificValue(x, typ(L)).value))
    # Finite-only loading does not decide scientific units of the formal input.
    loaded = GraphFunction.from_data(json.loads(json.dumps(function.to_data())))
    p, args = _formals(loaded.inputs)
    loaded(p, q=args["q"])
    with pytest.raises(TypeError, match="formal needs authoritative input meanings"):
        p.validate()
    with pytest.raises(TypeError, match="formal needs authoritative input meanings"):
        validate_scientific_program(p, strict=True, _allow_unbound_formals=True)
