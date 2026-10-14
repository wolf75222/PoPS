"""Scientific rules inspect real GraphFunction SSA; no numerical surrogate evaluator."""
from dataclasses import replace
import copy

import pytest

from pops._ir.quantity import PhysicalDimension, PhysicalSupport
from pops.model.spaces import Space
from pops.representations import ScientificMap, ScientificType, ScientificValue
from pops.scientific_graph import (ScientificConversion, check_scientific_contract_data,
                                   validate_scientific_program)
from pops.time import Program
from pops.time.expressions import ProgramScalar
from pops.time.variant_types import Float64, Product
from pops.graph_functions import graph
import pops.math as math


L = PhysicalDimension((("length", 1),))
T = PhysicalDimension((("time", 1),))
ONE = PhysicalDimension()


def typ(unit=ONE, *, kind="point_value", support=(), affine="linear", origin=None,
        finite=Float64, components=("q",), units=None, dofs=None, domain="real",
        sampling="point", representation="coordinates"):
    units = (unit,) if units is None else units
    shape = () if len(components) == 1 else (len(components),)
    space = Space("q", components, support=PhysicalSupport(support), units=units,
                  value_shape=shape, sampling=sampling, representation=representation,
                  domain=domain)
    return ScientificType(finite, space, kind, () if not shape else ("component",),
                          tuple(L for _ in support), {"name": "counting"}, dofs=dofs,
                          affine=affine, affine_origin=origin)


def checked(inputs, output, body, *, purpose=None):
    function = graph(inputs=Product(**{key: value.finite_type for key, value in inputs.items()}),
                     output=output.finite_type)(body)
    conversion = None if purpose is None else ScientificConversion(function.identity, purpose,
        "The exact finite graph defines this authored change of coordinates; no inverse theorem claimed.")
    return ScientificMap(function, inputs=inputs, output=output, strict=True, conversion=conversion)


def test_derived_units_and_exact_reloaded_contract():
    square = PhysicalDimension((("length", 2),))
    mapping = checked({"q": typ(L)}, typ(square), lambda p, x: x["q"] * x["q"])
    assert check_scientific_contract_data(mapping.checked_contract.to_data()).identity == mapping.checked_contract.identity
    p = Program("units")
    q = ScientificValue(p.float64(3), typ(L))
    result = mapping(p, q=q)
    assert result.scientific_type.space.units == (square,)
    assert validate_scientific_program(p, strict=True)
    assert "scientific_graph_v14" in str(p._serialize())


@pytest.mark.parametrize("body,source", [
    (lambda p, x: -x["q"], typ(domain="positive")),
    (lambda p, x: x["q"], typ()),
    (lambda p, x: p.float64(0), typ()),
    (lambda p, x: 0-x["q"], typ(domain="positive")),
])
def test_refined_domain_requires_an_explicit_obligation(body, source):
    with pytest.raises(TypeError, match="domain.*obligation"):
        checked({"q": source}, typ(domain="positive"), body)


@pytest.mark.parametrize("other", [lambda p, x: p.float64(0), lambda p, x: x["other"]])
def test_branch_unification_does_not_inherit_a_refined_domain(other):
    with pytest.raises(TypeError, match="domain.*obligation"):
        checked({"q": typ(domain="positive"), "other": typ()}, typ(domain="positive"),
                lambda p, x: p.branch(p.boolean(True), lambda b: x["q"], lambda b: other(b, x)))


@pytest.mark.parametrize("change", [dict(sampling="cell_average"), dict(representation="moments")])
def test_same_shape_does_not_convert_sampling_or_representation(change):
    source = typ(**change)
    with pytest.raises(TypeError, match="representation"):
        checked({"q": source}, typ(), lambda p, x: x["q"])
    explicit = checked({"q": source}, typ(), lambda p, x: x["q"], purpose="representation")
    assert check_scientific_contract_data(explicit.checked_contract.to_data())


def test_refined_domain_identity_preserves_the_exact_same_space():
    positive = typ(domain="positive")
    mapping = checked({"q": positive}, positive, lambda p, x: x["q"])
    p = Program("refined_identity")
    mapping(p, q=ScientificValue(p.float64(1), positive))
    assert validate_scientific_program(p, strict=True)
    assert mapping.checked_contract.to_data()["domain_obligations"] == []


@pytest.mark.parametrize("target", [
    typ(L, domain="positive"),
    typ(domain="positive", sampling="cell_average"),
    typ(domain="positive", representation="moments"),
    typ(domain="positive", support=(("x", "volume"),)),
])
def test_domain_assumption_does_not_exempt_units_sampling_representation_or_support(target):
    with pytest.raises(TypeError, match="dimension|support|representation"):
        checked({"q": typ()}, target, lambda p, x: x["q"], purpose="domain")


def test_representation_conversion_does_not_prove_domain_membership():
    with pytest.raises(TypeError, match="domain.*obligation"):
        checked({"q": typ()}, typ(domain="positive"), lambda p, x: x["q"], purpose="representation")


def test_domain_assumption_is_explicit_in_serialized_equations_and_rechecked():
    import json
    from pops.graph_functions import GraphFunction
    source, target = typ(), typ(domain="positive")
    mapping = checked({"q": source}, target, lambda p, x: x["q"], purpose="domain")
    contract = mapping.checked_contract.to_data()
    obligation, = contract["domain_obligations"]
    assert obligation == dict(classification="analytical_assumption",
        graph_identity=mapping.function.identity, justification=mapping.conversion.justification,
        inputs={"q": "real"}, output="positive")
    assert check_scientific_contract_data(json.loads(json.dumps(contract)))
    # The assumption is recorded on the actual closed Program graph, not just on
    # a discarded Python wrapper. Reloading rechecks it but does not prove it.
    wrapped = graph(inputs=Product(q=Float64), output=Float64)(lambda p, x:
        mapping(p, q=ScientificValue(x["q"], source)).value)
    serialized = json.loads(json.dumps(wrapped.to_data()))
    assert "analytical_assumption" in json.dumps(serialized)
    loaded = GraphFunction.from_data(serialized)
    p = Program("reloaded_domain_assumption")
    loaded(p, q=p.float64(1))
    assert validate_scientific_program(p, strict=True)
    forged = copy.deepcopy(contract)
    forged["domain_obligations"] = []
    with pytest.raises(TypeError, match="metadata differs"):
        check_scientific_contract_data(forged)


def test_remap_validates_actual_candidate_meaning_without_reusing_old_annotation():
    from pops.bundles import BundleSchema
    from pops.time._program.structured_control import new_typed
    from pops.time.variant_types import type_of
    p=Program('scientific_remap_candidate')
    domain=p.range_set(2,capacity=2,ownership='replicated')
    old_raw=p.map(domain,lambda b,i:b.float64(1))
    new_raw=p.map(domain,lambda b,i:b.float64(2))
    old_type=typ(L,finite=type_of(old_raw),dofs=type_of(domain))
    new_type=typ(T,finite=type_of(new_raw),dofs=type_of(domain))
    old_q=ScientificValue(old_raw,old_type).value
    new_q=ScientificValue(new_raw,new_type).value
    def packet(q):
        return p.product(instance=domain,fields=p.product(q=q),geometry=p.product(),
                         maps=p.product(source=domain),histories=p.product())
    raw=packet(old_q)
    schema=BundleSchema(raw,domains={'current':('instance',),'source':('maps','source')},
                        bindings={('fields','q'):'current'})
    old=schema.bind(p,raw);candidate=schema.bind(p,packet(new_q))
    checked=new_typed(p,type_of(old).signature,'bundle_remap_validate',(old,candidate),
                     {'source_instance':('maps','source'),'history_policies':()},point=candidate.point)
    output=checked['fields']['q']
    # Returning old input x[0] would wrongly let this claim of length pass.
    with pytest.raises(TypeError,match='dimension|meaning'):
        ScientificValue(output,old_type)
    assert ScientificValue(output,new_type).scientific_type==new_type


@pytest.mark.parametrize("body,inputs,output", [
    (lambda p, x: x["a"] + x["b"], {"a": typ(L), "b": typ(T)}, typ(L)),
    (lambda p, x: math.exp(x["a"]), {"a": typ(L)}, typ()),
    (lambda p, x: math.log(x["a"]), {"a": typ(T)}, typ()),
    (lambda p, x: x["a"] * x["a"], {"a": typ(L)}, typ(L)),
])
def test_dimension_errors_cannot_be_fixed_by_output_annotation(body, inputs, output):
    with pytest.raises(TypeError, match="dimension"):
        checked(inputs, output, body)


def test_component_dimensions_checked_independently():
    finite = Product(length=Float64, time=Float64)
    mixed = typ(finite=finite, components=("length", "time"), units=(L, T))
    checked({"q": mixed}, mixed,
            lambda p, x: p.product(length=2*x["q"]["length"], time=x["q"]["time"]/3))
    with pytest.raises(TypeError, match="dimensions"):
        checked({"q": mixed}, mixed,
                lambda p, x: p.product(length=x["q"]["time"], time=x["q"]["length"]))


def test_affine_difference_is_increment_and_affine_sum_is_rejected():
    point = typ(L, affine="point", origin="coordinate_origin")
    increment = replace(point, affine="increment")
    checked({"a": point, "b": point}, increment, lambda p, x: x["a"]-x["b"])
    checked({"a": point, "d": increment}, point, lambda p, x: x["a"]+x["d"])
    with pytest.raises(TypeError, match="two affine points"):
        checked({"a": point, "b": point}, point, lambda p, x: x["a"]+x["b"])
    with pytest.raises(TypeError, match="affine"):
        checked({"a": point, "b": point}, point, lambda p, x: x["a"]-x["b"])


def test_support_and_primal_dual_are_not_same_shape_conversions():
    volume = typ(L, support=(("x", "volume"),))
    surface = typ(L, support=(("s", "surface"),))
    with pytest.raises(TypeError, match="support"):
        checked({"a": volume, "b": surface}, volume, lambda p, x: x["a"]+x["b"])
    primal, dual = typ(kind="coefficient"), typ(kind="dual_charge")
    with pytest.raises(TypeError, match="representation"):
        checked({"a": primal, "b": dual}, primal, lambda p, x: x["a"]+x["b"])
    with pytest.raises(TypeError, match="representation"):
        checked({"a": primal}, dual, lambda p, x: 2*x["a"])


def test_explicit_finite_conversion_keeps_unit_checks_and_exact_body_identity():
    physical, coefficient = typ(L), typ(L, kind="coefficient")
    mapping = checked({"q": physical}, coefficient, lambda p, x: 2*x["q"], purpose="representation")
    p = Program("conversion")
    result = mapping(p, q=ScientificValue(p.float64(1), physical))
    assert result.scientific_type.kind == "coefficient"
    with pytest.raises(TypeError, match="dimensions"):
        checked({"q": physical}, coefficient, lambda p, x: x["q"]*x["q"], purpose="representation")
    other = graph(inputs=Product(q=Float64), output=Float64)(lambda p, x: x["q"]/2)
    with pytest.raises(TypeError, match="different finite graph"):
        ScientificMap(other, inputs={"q": physical}, output=coefficient,
                      strict=True, conversion=mapping.conversion)


def test_scientific_type_survives_value_extraction_and_canonical_ssa_alias():
    p = Program("aliases")
    raw = p.float64(3)
    value = ScientificValue(raw, typ(L))
    for alias in (raw, value.value):
        with pytest.raises(TypeError, match="reannotation"):
            ScientificValue(alias, typ(T))
    with pytest.raises(TypeError, match="dimensions"):
        ScientificValue(value.value*value.value, typ(L))


def test_branch_and_fold_check_scientific_carried_type():
    checked({"q": typ(L)}, typ(L),
            lambda p, x: p.fold(x["q"], 3, lambda b, y, i: y*2))
    with pytest.raises(TypeError, match="dimensions"):
        checked({"q": typ(L)}, typ(L),
                lambda p, x: p.fold(x["q"], 3, lambda b, y, i: y*y))
    with pytest.raises(TypeError, match="dimensions"):
        checked({"a": typ(L), "b": typ(T)}, typ(L),
                lambda p, x: p.branch(p.boolean(True), lambda b: x["a"], lambda b: x["b"]))


def test_forged_serialized_output_does_not_turn_square_into_length():
    mapping = checked({"q": typ(L)}, typ(PhysicalDimension((("length", 2),))),
                      lambda p, x: x["q"]*x["q"])
    changed = copy.deepcopy(mapping.checked_contract.to_data())
    changed["output"] = typ(L).to_data()
    with pytest.raises(TypeError, match="dimensions"):
        check_scientific_contract_data(changed)


def test_legacy_map_is_explicitly_not_normative():
    function = graph(inputs=Product(q=Float64), output=Float64)(lambda p, x: 2*x["q"])
    legacy = ScientificMap(function, inputs={"q": typ()}, output=typ(kind="coefficient"))
    assert legacy.to_data()["checking"] == "legacy_annotations"
    p = Program("legacy")
    legacy(p, q=ScientificValue(p.float64(1), typ()))
    with pytest.raises(TypeError, match="legacy"):
        validate_scientific_program(p, strict=True)


def test_array_diffusion_checks_coordinate_and_diffusivity_units_on_actual_graph():
    from pops.time.variant_types import type_of
    p = Program("array_diffusion")
    d = p.range_set(4, capacity=4, ownership="replicated")
    q = p.map(d, lambda b, i: b.float64(b.index_value(i)))
    array = type_of(q)
    domain = type_of(d)
    support = (("x", "periodic_line"),)
    qtype = typ(L, finite=array, dofs=domain, support=support)
    dx = typ(L, support=support)
    diffusivity = typ(PhysicalDimension((("length", 2), ("time", -1))), support=support)
    rate = typ(PhysicalDimension((("length", 1), ("time", -1))),
               finite=array, dofs=domain, support=support)
    def body(b, x):
        domain = b.domain(x["q"])
        n = b.size(domain)
        def cell(c, i):
            j = c.index_value(i)
            left = c.at(x["q"], c.index(domain, (j+n-1) % n))
            right = c.at(x["q"], c.index(domain, (j+1) % n))
            return x["k"]*(left-2*c.at(x["q"], i)+right)/(x["dx"]*x["dx"])
        return b.map(domain, cell)
    inputs = {"q": qtype, "k": diffusivity, "dx": dx}
    mapping = checked(inputs, rate, body)
    actual = mapping(p, q=ScientificValue(q, qtype),
                     k=ScientificValue(p.float64(.1), diffusivity),
                     dx=ScientificValue(p.float64(.25), dx))
    assert actual.value.op == "finite_map"
    assert validate_scientific_program(p, strict=True)
    assert check_scientific_contract_data(mapping.checked_contract.to_data(),
                                         inputs=inputs, output=rate).identity == mapping.checked_contract.identity
    with pytest.raises(TypeError, match="dimensions"):
        checked(dict(inputs, dx=typ(T, support=support)), rate, body)


def test_scientific_dependency_without_output_annotation_is_checked():
    p = Program("erased_annotation")
    q = ScientificValue(p.float64(2), typ(L))
    math.exp(q.value)
    with pytest.raises(TypeError, match="dimensionless"):
        validate_scientific_program(p, strict=True)


def test_applied_contract_rejects_modified_numerical_body():
    from pops.scientific_graph import CONTRACT
    mapping = checked({"q": typ(L)}, typ(PhysicalDimension((("length", 2),))),
                      lambda p, x: x["q"]*x["q"])
    p = Program("tampered_body")
    value = mapping(p, q=ScientificValue(p.float64(2), typ(L))).value
    canonical = p._canonical_value(value)
    assert CONTRACT in canonical.attrs
    p._replace_value(canonical, attrs=dict(canonical.attrs, fn="add"))
    with pytest.raises(TypeError, match="different applied body"):
        validate_scientific_program(p, strict=True)


def test_region_capture_cannot_be_reannotated_as_an_external_datum():
    from pops.time.variant_types import type_of
    p = Program("region_escape")
    d = p.range_set(3, capacity=3, ownership="replicated")
    length = ScientificValue(p.float64(2), typ(L))
    squares = p.map(d, lambda b, i: length.value*length.value)
    with pytest.raises(TypeError, match="dimensions"):
        ScientificValue(squares, typ(L, finite=type_of(squares), dofs=type_of(d)))


def test_scalar_expression_comparison_preserves_input_dimensions():
    checked({"a": typ(L)}, typ(L), lambda p, x: p.scalar_expression("where", math.where(ProgramScalar(x["a"]) > 0,
            lambda: math.sqrt(ProgramScalar(x["a"])*ProgramScalar(x["a"])), lambda: -ProgramScalar(x["a"]))))
    with pytest.raises(TypeError, match="dimensions"):
        checked({"a": typ(L), "b": typ(T)}, typ(L), lambda p, x:
                p.scalar_expression("where_bad", math.where(ProgramScalar(x["a"]) > ProgramScalar(x["b"]),
                    lambda: ProgramScalar(x["a"]), lambda: -ProgramScalar(x["a"]))))


def test_continuation_schema_preserves_scientific_meaning_across_read_and_freeze():
    from pops.continuations import ContinuationSchema
    from pops.time.variant_types import type_of
    p = Program("persistent_units")
    domain = p.range_set(3, capacity=3, ownership="replicated")
    q = p.map(domain, lambda b, i:b.float64(2))
    qtype = typ(L, finite=type_of(q), dofs=type_of(domain))
    time_type = typ(T)
    initial = p.product(q=q, time=p.float64(0), step=p.int64(0))
    schema = ContinuationSchema(initial, fields=(), time=('time',), step=('step',),
        scientific={('q',):qtype, ('time',):time_type})
    base = p.continuation_open(initial, schema)
    current = p.continuation_read(base)
    with pytest.raises(TypeError, match="dimensions"):
        ScientificValue(current['q'], replace(qtype, space=typ(T).space))
    h = ScientificValue(p.float64(.1), time_type).value
    def envelope(values):
        return p.product(continuation=p.product(q=values, time=current['time']+h,
                         step=current['step']+1), increment=p.product(), observations=p.product(), checks=p.product())
    good = p.map(domain, lambda b, i:2*b.at(current['q'], i))
    p.continuation_freeze(base, p.product(h=h), envelope(good))
    wrong = p.map(domain, lambda b, i:b.at(current['q'], i)**2)
    with pytest.raises(TypeError, match="dimensions"):
        p.continuation_freeze(base, p.product(h=h), envelope(wrong))
    assert validate_scientific_program(p, strict=True)


def test_continuation_scientific_schema_covers_nested_arrays_and_rejects_tampered_units():
    from pops.continuations import ContinuationSchema
    from pops.time.variant_types import type_of
    from pops.scientific_graph import validate_scientific_packet_data
    p = Program("nested_scientific_packet")
    outer = p.range_set(2, capacity=2, ownership="replicated")
    inner = p.range_set(3, capacity=3, ownership="replicated")
    values = p.map(outer, lambda b, i:b.map(inner, lambda c, j:c.float64(1)))
    qtype = typ(L, finite=type_of(values), dofs=type_of(outer))
    initial = p.product(fields=p.product(q=values), time=p.float64(0), step=p.int64(0))
    with pytest.raises(TypeError, match="does not cover"):
        ContinuationSchema(initial, fields=(), time=('time',), step=('step',),
                           scientific={('fields','q'):qtype})
    schema = ContinuationSchema(initial, fields=(), time=('time',), step=('step',),
        scientific={('fields','q'):qtype, ('time',):typ(T)})
    validate_scientific_packet_data(schema.to_data())
    import json
    from pops.scientific_graph import _key
    data = dict(schema.to_data())
    data['scientific'] = json.loads(_key(data['scientific']))
    data['scientific'][0]['type']['space']['units'] = [T.to_data()]
    with pytest.raises(TypeError, match="identity changed"):
        validate_scientific_packet_data(data)


def test_continuation_paths_type_array_elements_and_inactive_variant_payloads():
    from pops.continuations import ContinuationSchema
    from pops.time.variant_types import Variant
    from pops.scientific_graph import validate_scientific_packet_data
    p = Program('scientific_array_variant_paths')
    domain = p.range_set(2, capacity=2, ownership='replicated')
    payload = Variant(distance=Float64, duration=Float64)
    values = p.map(domain, lambda b,i:b.product(
        position=b.float64(2), event=b.variant(payload, 'distance', b.float64(3))))
    initial = p.product(q=values, time=p.float64(0), step=p.int64(0))
    declarations = {('q','[]','position'):typ(L),
                    ('q','[]','event','distance'):typ(L),
                    ('q','[]','event','duration'):typ(T), ('time',):typ(T)}
    schema = ContinuationSchema(initial, time=('time',), step=('step',), scientific=declarations)
    validate_scientific_packet_data(schema.to_data())
    base = p.continuation_open(initial, schema)
    current = p.continuation_read(base)
    h = ScientificValue(p.float64(.1), typ(T)).value
    def updates(b,i):
        previous = b.at(current['q'], i)
        # The duration arm is initially inactive, but its declared meaning must
        # already survive read/match and be checked if it later becomes active.
        distance = b.match(previous['event'], distance=lambda c,x:x,
            duration=lambda c,x:previous['position'])
        return b.product(position=distance, event=b.variant(payload,'duration',h))
    q = p.map(domain, updates)
    def envelope(q):
        return p.product(continuation=p.product(q=q, time=current['time']+h,
            step=current['step']+1), increment=p.product(), observations=p.product(), checks=p.product())
    p.continuation_freeze(base, p.product(h=h), envelope(q))
    bad = p.map(domain, lambda b,i:b.product(position=b.at(current['q'],i)['position'],
        event=b.variant(payload,'duration',b.at(current['q'],i)['position'])))
    with pytest.raises(TypeError, match='dimensions'):
        p.continuation_freeze(base,p.product(h=h),envelope(bad))
    assert validate_scientific_program(p, strict=True)


@pytest.mark.parametrize('region', ('root','branch','map'))
def test_unused_checked_operations_cannot_hide_dimension_errors(region):
    def body(p,a):
        def wrong(b):
            _unused = a['length'] + a['time']
            return a['length']
        if region=='root':return wrong(p)
        if region=='branch':return p.branch(p.boolean(True),wrong,lambda b:a['length'])
        domain=p.range_set(2,capacity=2,ownership='replicated')
        _unused_array=p.map(domain,lambda b,i:wrong(b))
        return a['length']
    with pytest.raises(TypeError,match='dimensions'):
        checked({'length':typ(L),'time':typ(T)},typ(L),body)


def test_plain_graph_call_checks_unused_scientific_operations_atomically():
    @graph(inputs=Product(length=Float64,time=Float64),output=Float64)
    def function(p,a):
        _unused=a['length']+a['time']
        return a['length']
    p=Program('plain_graph_scientific_call')
    length=ScientificValue(p.float64(2),typ(L)).value
    duration=ScientificValue(p.float64(3),typ(T)).value
    before=tuple(p._values)
    with pytest.raises(TypeError,match='dimensions'):
        function(p,length=length,time=duration)
    assert tuple(p._values)==before


def test_bundle_reconstruction_cannot_relabel_scientific_history():
    from pops.bundles import BundleSchema
    from pops.continuations import ContinuationSchema
    from pops.time.variant_types import type_of
    p=Program('scientific_bundle_history')
    d=p.range_set(2,capacity=2,ownership='replicated')
    q=p.map(d,lambda b,i:b.float64(2))
    def raw(values,history):
        return p.product(instance=d,fields=p.product(q=values),geometry=p.product(),
                         maps=p.product(),histories=p.product(previous=history))
    bundle_type=BundleSchema(raw(q,q),domains={'current':('instance',)},
        bindings={('fields','q'):'current',('histories','previous'):'current'})
    initial=p.product(bundle=bundle_type.bind(p,raw(q,q)),time=p.float64(0),step=p.int64(0))
    qtype=typ(L,finite=type_of(q),dofs=type_of(d))
    schema=ContinuationSchema(initial,time=('time',),step=('step',),scientific={
        ('bundle','fields','q'):qtype,('bundle','histories','previous'):qtype,('time',):typ(T)})
    base=p.continuation_open(initial,schema);old=p.continuation_read(base)
    prior=old['bundle']['fields']['q']
    wrong=p.map(d,lambda b,i:b.at(prior,i)**2)
    bundle=bundle_type.bind(p,raw(prior,wrong))
    candidate=p.product(continuation=p.product(bundle=bundle,time=old['time'],step=old['step']),
                        increment=p.product(),observations=p.product(),checks=p.product())
    with pytest.raises(TypeError,match='dimensions'):
        p.continuation_freeze(base,p.product(),candidate)
