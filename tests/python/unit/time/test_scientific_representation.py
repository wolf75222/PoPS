"""Scientific compatibility and actual finite-graph composition, source only."""
from dataclasses import replace

import pytest

from pops._ir.quantity import PhysicalDimension, PhysicalSupport
from pops.model.spaces import Space
from pops.representations import (Representation, ScientificType, ScientificValue,
                                  ScientificMap, require_scientific_output)
from pops.time import Program
from pops.time.variant_types import Float64, Int64, Product, type_of


def _type(kind="point_value", *, measure=None, finite_type=Float64, dofs=None):
    space = Space("physical_q", ("q",), value_shape=(),
                  support=PhysicalSupport(()), units=(PhysicalDimension(),),
                  representation="physical_quantity", sampling="point")
    return ScientificType(finite_type, space, kind, (), (),
                          {"name": "counting"} if measure is None else measure, dofs)


def _map(inputs, output, builder):
    from pops.graph_functions import graph
    function = graph(inputs=Product(**{key: value.finite_type for key, value in inputs.items()}),
                     output=output.finite_type)(builder)
    return ScientificMap(function, inputs=inputs, output=output)


def _representation():
    physical, coefficient, dual = _type(), _type("coefficient"), _type("dual_charge")
    encode = _map({"value": physical}, coefficient,
                  lambda p, x: p.value("twice", 2*x["value"]))
    # Deliberately not an inverse: type compatibility must not invent that theorem.
    decode = _map({"value": coefficient}, physical,
                  lambda p, x: p.value("third", x["value"]/3))
    pairing = _map({"left": coefficient, "right": dual}, _type("pairing_value"),
                   lambda p, x: p.value("weighted_pairing", 5*x["left"]*x["right"]))
    return Representation(physical.space, dofs=None, kind="coefficient",
                          encode=encode, decode=decode, pairing=pairing)


def test_executable_maps_create_native_ssa_and_preserve_equations():
    from pops.codegen.program_codegen import emit_cpp_program
    from typed_program_support import typed_state
    from pops.model import StateSpace
    p = Program("scientific_maps")
    state = typed_state(p, "block", state_name="U",
                        space=StateSpace('U', ('U',), units=(PhysicalDimension(),)))
    state.n
    rep = _representation()
    raw = ScientificValue(p.float64(3.), rep.encode.inputs["value"])
    encoded = rep.encode(p, value=raw)
    decoded = rep.decode(p, value=encoded)
    dual = ScientificValue(p.float64(7.), rep.pairing.inputs["right"])
    paired = rep.pairing(p, left=encoded, right=dual)
    # Keep the heterogeneous results separate. Both finite maps are consumed by
    # the native expression, without adding unlike scientific representations.
    decoded_positive = decoded.value > 0
    paired_positive = paired.value > 0
    factor = p.branch(decoded_positive,
        lambda b: b.branch(paired_positive, lambda c: c.float64(1), lambda c: c.float64(0)),
        lambda b: b.float64(0))
    p.commit(state.next, p.value("accepted", state.n*factor, at=state.next.point))
    assert p.validate()
    cpp = emit_cpp_program(p)
    assert "pops::Real(3)" in cpp and "pops::Real(5)" in cpp
    assert " / " in cpp and " * " in cpp
    assert "ctx.commit_many(" in cpp
    assert encoded.scientific_type.kind == "coefficient"
    assert decoded.scientific_type.kind == "point_value"
    assert rep.to_dict()["data"]["encode"]["function"]


@pytest.mark.parametrize("change", [
    {"kind": "dual_charge"}, {"measure": {"name": "weighted_counting"}},
    {"space": Space("physical_q", ("q",), value_shape=(), support=PhysicalSupport(()),
                    units=(PhysicalDimension((("length", 1),)),),
                    representation="physical_quantity", sampling="point")},
])
def test_equal_machine_shape_does_not_imply_scientific_compatibility(change):
    original = _type("coefficient")
    foreign = replace(original, **change)
    p = Program("bad_scientific_type")
    value = ScientificValue(p.float64(1), original)
    with pytest.raises(TypeError, match="support, units, variance"):
        ScientificValue(value, foreign)
    with pytest.raises(TypeError, match="realized output"):
        require_scientific_output(value, foreign, p)


def test_unknown_units_support_and_variance_are_not_silently_filled():
    unknown = Space("q", ("q",), value_shape=())
    with pytest.raises(ValueError, match="explicit support"):
        replace(_type(), space=unknown)
    vector = Space("vector", ("x", "y"), support=PhysicalSupport(()),
                   units=(PhysicalDimension(), PhysicalDimension()))
    with pytest.raises(ValueError, match="variance"):
        replace(_type(), space=vector)
    contra = replace(_type(), space=vector, variance=("contravariant",))
    covar = replace(contra, variance=("covariant",))
    with pytest.raises(TypeError, match="variance"):
        contra.require_compatible(covar)


def test_array_scientific_type_retains_exact_dof_family():
    p = Program("scientific_array")
    first = p.range_set(2, capacity=3, ownership="replicated")
    second = p.range_set(2, capacity=3, ownership="replicated")
    values = p.map(first, lambda b, i: b.float64(b.index_value(i)))
    good = _type(finite_type=type_of(values), dofs=type_of(first))
    assert ScientificValue(values, good).value is values
    with pytest.raises(ValueError, match="dof family disagree"):
        replace(good, dofs=type_of(second))


def test_map_rejects_opaque_body_wrong_output_and_untyped_binding():
    from pops.graph_functions import graph
    typ = _type()
    with pytest.raises(TypeError, match="closed GraphFunction"):
        ScientificMap(lambda p, x: x, inputs={"value": typ}, output=typ)
    fn = graph(inputs=Product(value=Float64), output=Int64)(lambda p, x: p.int64(3))
    with pytest.raises(TypeError, match="output annotation"):
        ScientificMap(fn, inputs={"value": typ}, output=typ)
    rep = _representation()
    p = Program("bad_binding")
    raw = p.float64(1)
    before = p._serialize()
    with pytest.raises(TypeError, match="ScientificValue"):
        rep.encode(p, value=raw)
    assert p._serialize() == before


def test_pairing_requires_declared_dual_and_owner_before_erasure():
    rep = _representation()
    p, other = Program("owner"), Program("foreign")
    left = ScientificValue(p.float64(1), rep.pairing.inputs["left"])
    wrong = ScientificValue(p.float64(2), rep.pairing.inputs["left"])
    with pytest.raises(TypeError, match="realized output"):
        rep.pairing(p, left=left, right=wrong)
    foreign = ScientificValue(other.float64(2), rep.pairing.inputs["right"])
    with pytest.raises((ValueError, TypeError), match="owned|program|Program|belong"):
        rep.pairing(p, left=left, right=foreign)


def test_coefficients_and_dual_are_qualified_by_the_actual_finite_maps():
    from examples.api_v02.representation_checks import finite_representation
    a = finite_representation(encode_scale=2.)
    a_again = finite_representation(encode_scale=2.)
    b = finite_representation(encode_scale=3.)
    assert a == a_again
    assert a.encode.output == a_again.encode.output
    assert a.encode.output != b.encode.output
    assert a.pairing.inputs["right"] != b.pairing.inputs["right"]
    p = Program("different_bases")
    physical = ScientificValue(p.float64(2.), a.encode.inputs["value"])
    left_a = a.encode(p, value=physical)
    left_b = b.encode(p, value=physical)
    dual_a = ScientificValue(p.float64(3.), a.pairing.inputs["right"])
    a_again.decode(p, value=left_a)
    before = p._serialize()
    with pytest.raises(TypeError, match="representation"):
        b.decode(p, value=left_a)
    with pytest.raises(TypeError, match="representation"):
        b.pairing(p, left=left_b, right=dual_a)
    assert p._serialize() == before
    with pytest.raises(TypeError, match="unqualified"):
        Representation(a.space, dofs=a.dofs, kind=a.kind,
                       encode=a.encode, decode=a.decode, pairing=a.pairing)


def test_legacy_representation_identity_and_immutability_are_unchanged():
    first = Representation("conservative", {"weights": [1., 2.]})
    assert first.to_dict() == {"schema_version": 1, "category": "representation",
                               "name": "conservative", "data": {"weights": [
                                   {"binary64": "0x1.0000000000000p+0"},
                                   {"binary64": "0x1.0000000000000p+1"}]}}
    with pytest.raises((AttributeError, TypeError)):
        first.data["weights"] = (7.,)
    with pytest.raises(TypeError, match="physical Space"):
        Representation("conservative", encode=lambda x: x)


def test_realize_checks_scientific_output_before_erasure_and_exports_claims():
    import pops
    from pops.claims import Claim
    from pops.frames import Cartesian1D
    from pops.math import ddt
    from pops.realization import Realize, Use
    from typed_program_support import state_refs
    model = pops.Model("representation_realize", frame=Cartesian1D())
    state = model.state("q", components=("q",))
    source = model.source("production", on=state, value=(state[0],))
    rate = model.rate("balance", equation=ddt(state) == source)
    p = Program("representation_realize")
    block, _ = state_refs(p, "material", model=model, state=state)
    q = p.state(block[state])
    q.n
    rep = _representation()
    wanted = rep.encode.output
    use = Use(rate.occurrences[0], "cell", "current_stage", "coefficient", scientific_type=wanted)
    raw = ScientificValue(p.float64(3), rep.encode.inputs["value"])
    assumption = Claim.assumption("finite projection models the requested quantity",
                                  scope="declared space", justification="explicit author assertion")
    recipe = Realize("projection", owns=(use,), bindings={"x": raw},
                     body=lambda b, x: rep.encode(b, value=x), outputs={use: ()},
                     claims=(assumption,))
    result = recipe.evaluate(p)[use]
    p.commit(q.next, p.value("accepted", q.n*result, at=q.next.point))
    data = p._serialize()
    text = str(data)
    assert "analytical_assumption" in text and "pops.scientific-type@1" in text
    wrong = replace(wanted, measure={"name": "different_measure"})
    foreign_use = replace(use, scientific_type=wrong)
    bad = Realize("incompatible", owns=(foreign_use,), bindings={"x": raw},
                  body=lambda b, x: rep.encode(b, value=x), outputs={foreign_use: ()})
    before = p._serialize()
    with pytest.raises(TypeError, match="realized output"):
        bad.evaluate(p)
    assert p._serialize() == before


def test_native_reception_builders_change_generated_equations_and_reject_stationary_state():
    import pops
    from examples.api_v02.representation_checks import build_representation_check
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    sources, hashes = [], []
    for scales in ((2., 3., 5.), (3.5, 4., .75)):
        spec = build_representation_check(encode_scale=scales[0], decode_scale=scales[1],
                                          pairing_scale=scales[2])
        plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
        sources.append(emit_cpp_program(plan.time,
            model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks)))
        hashes.append(plan.time._ir_hash())
        assert not spec.invariants({"samples": next(iter(spec.initial_values.values()))})["passed"]
    assert sources[0] != sources[1]
    assert hashes[0] != hashes[1]
    assert all("ctx.commit_many(" in source for source in sources)


def _resolved_representation(scales=(2., 3., 5.)):
    import pops
    from examples.api_v02.representation_checks import build_representation_check
    spec = build_representation_check(encode_scale=scales[0], decode_scale=scales[1],
                                      pairing_scale=scales[2])
    return pops.resolve(pops.validate(spec.case), layout=spec.layout)


@pytest.mark.parametrize("scales", ((2., 3., 5.), (3.5, 4., .75)))
def test_scientific_maps_survive_actual_compiler_detachment(scales):
    """Follow Program compiler preparation; no native import, compiler or DSO substitute."""
    from pops.codegen._orchestration_compile import build_program_model_graph
    from pops.codegen.program_models import prepare_program_authority
    from pops.codegen._shared_interface_evidence import _issue_shared_interface_codegen_evidence
    from pops.codegen.program_graph_lowering import _emit_resolved_program_graph
    from pops.codegen.program_balance_due import validate_balance_due_contract
    from pops._balance_due_contract import BalanceDueContract
    from pops.time._program.detach import detach_compiled_program
    from pops.scientific_graph import validate_scientific_program
    plan = _resolved_representation(scales)
    model_graph = build_program_model_graph(plan)
    prepare_program_authority(None, model_graph)
    contract = BalanceDueContract.from_consumer_graph(plan.consumer_graph)
    validate_balance_due_contract(plan.time, contract)
    source_image = plan.time._serialize(include_provenance=False)
    detached = detach_compiled_program(plan.time,
        physical_global_sources=plan._physical_global_sources)
    assert detached._serialize(include_provenance=False) == source_image
    assert validate_scientific_program(detached, strict=True)
    # Redetachment is a verified internal rebind, never a copied authoring registry.
    redetached = detach_compiled_program(detached)
    assert redetached._ir_hash() == detached._ir_hash() == plan.time._ir_hash()
    assert validate_scientific_program(redetached, strict=True)
    assert detached._scientific_compile_binding.program is detached
    proof = detached._scientific_compile_binding.proof
    assert isinstance(proof.image, str)
    assert all(type(identifier) is int and type(data) is str
               for identifier, data in proof.contracts)
    for node in detached._values:
        if node.block is not None:
            assert node.block._instance_registry is None
    cpp = _emit_resolved_program_graph(detached.to_graph(), lowering_program=detached,
        model=None, model_graph=model_graph, target=plan.target, field_plans=plan.field_plans,
        balance_due_contract=contract,
        shared_interface_codegen_evidence=_issue_shared_interface_codegen_evidence(plan))
    assert "ctx.commit_many(" in cpp
    assert "pops_program_abi_key" in cpp
    from pops.codegen.program_source_map import source_map
    assert source_map(cpp)["complete"]


@pytest.mark.parametrize("corruption", ("missing", "copied", "snapshot", "node_identity"))
def test_detached_scientific_proof_is_not_a_reusable_or_mutable_token(corruption):
    from pops.time._program.detach import detach_compiled_program
    from pops.scientific_graph import validate_scientific_program, CONTRACT
    plan = _resolved_representation()
    detached = detach_compiled_program(plan.time)
    if corruption == "missing":
        object.__delattr__(detached, "_scientific_compile_binding")
    elif corruption == "copied":
        other = detach_compiled_program(plan.time)
        object.__setattr__(detached, "_scientific_compile_binding", other._scientific_compile_binding)
    elif corruption == "snapshot":
        object.__setattr__(detached, "name", "different_program")
    else:
        node = next(value for value in detached._values if CONTRACT in value.attrs)
        duplicate = type(node)(node.prog, node.id, node.vtype, node.op, node.inputs,
            node.attrs, node.name, node.block, space=node.space,
            source_location=node.source_location, field_context=node.field_context,
            region=node.region, state_ref=node.state_ref, point=node.point,
            provenance=node.provenance)
        assert duplicate is not node
        object.__setattr__(detached, "_values", tuple(
            duplicate if value is node else value for value in detached._values))
        # Equal bytes/IDs are deliberately insufficient for a bound native proof.
        assert detached._serialize(include_provenance=False) == plan.time._serialize(include_provenance=False)
    with pytest.raises((ValueError, TypeError), match="scientific.*(proof|validated|bound)|bound.*node"):
        validate_scientific_program(detached, strict=True)


def test_detached_scientific_block_remains_invalid_for_public_authoring():
    from pops.time._program.detach import detach_compiled_program
    from pops.time.references import bind_program_block
    from pops.graph_functions import GraphFunction
    from pops.scientific_graph import CONTRACT
    plan = _resolved_representation()
    detached = detach_compiled_program(plan.time)
    node = next(value for value in detached._values if CONTRACT in value.attrs)
    p = Program("foreign_author")
    before = p._serialize()
    with pytest.raises(ValueError, match="detached from its authoritative Case registry"):
        bind_program_block(p, node.block, where="negative scientific authoring")
    contract = node.attrs[CONTRACT]
    # Public reload does not acquire a compiler proof from serialized metadata.
    from pops.representations import ScientificType
    inputs = {key: ScientificType.from_data(data, finite_type=type_of(child), space=child.space)
              for (key, data), child in zip(contract["inputs"].items(),
                  node.attrs["scientific_application_v14"]["input_values"].values(), strict=True)}
    with pytest.raises(ValueError, match="detached from its authoritative Case registry"):
        import json
        from pops.scientific_graph import _key
        GraphFunction.from_data(json.loads(_key(contract["graph"])),
            inputs=Product(**{key: item.finite_type for key, item in inputs.items()}),
            output=type_of(node))
    assert p._serialize() == before


def test_scientific_detachment_does_not_authenticate_a_foreign_live_case():
    from pops.time.references import bind_program_block
    first, second = _resolved_representation(), _resolved_representation()
    p = Program("same_name_distinct_case")
    block_a = next(value.block for value in first.time._values if value.block is not None)
    block_b = next(value.block for value in second.time._values if value.block is not None)
    bind_program_block(p, block_a, where="first")
    before = p._serialize()
    with pytest.raises(ValueError, match="distinct Case authoring authorities"):
        bind_program_block(p, block_b, where="foreign")
    assert p._serialize() == before


def _nested_scientific_plan(kind):
    """Strict dependent scalar maps live inside real region bodies, not dead metadata."""
    import pops
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.graph_functions import graph
    from pops.initial import InitialCondition
    from pops.layouts import Uniform
    from pops.lib.initial import BindArray
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, StateStorage
    from pops.projection import ConservativeCellAverage
    from pops.time import FixedDt
    from pops.time.expressions import ProgramScalar

    frame = Rectangle("nested_grid", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("nested_scientific", frame=frame)
    state = model.state("q", components=("q",), units=(PhysicalDimension(),))
    case = pops.Case("nested_scientific_" + kind)
    block = case.block("samples", model)
    numerics = DiscretizationPlan()
    numerics.states.add(state, StateStorage())
    case.numerics(numerics, block=block)
    p = Program("nested_scientific_" + kind)
    q = p.state(block[state])
    measured = p.norm_inf(q.n)
    finite = type_of(measured)
    scientific = _type(finite_type=finite)
    measured = ScientificValue(measured, scientific).value
    function = graph(inputs=Product(value=finite), output=finite)(
        lambda g, a: g.value("strict_identity", a["value"] + 0.))
    mapping = ScientificMap(function, inputs={"value": scientific},
                            output=scientific, strict=True)

    def apply(g, value):
        return mapping(g, value=ScientificValue(value, scientific)).value

    if kind == "branch_fold":
        result = p.branch(measured > 0,
            lambda b: b.fold(measured, 2, lambda g, carry, index: apply(g, carry)),
            lambda b: b.fold(measured, 1, lambda g, carry, index: apply(g, carry)))
    elif kind == "map_fold":
        domain = p.range_set(2, capacity=2, ownership="replicated")
        values = p.map(domain,
            lambda b, index: b.fold(measured, 2, lambda g, carry, step: apply(g, carry)))
        result = values[p.index(domain, 0)]
    elif kind == "relation_residual":
        result = _strict_residual_relation(p, measured, apply)
    else:
        raise AssertionError(kind)
    p.commit(q.next, p.value("accepted", (q.n[0] + ProgramScalar(result),), at=q.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 4), periodic=PeriodicAxes(frame.axes)))
    return pops.resolve(pops.validate(case), layout=layout)


def _strict_residual_relation(p, measured, apply):
    """An actual solve wrapper rechecks a strict original residual in its capture."""
    from pops.graph_functions import graph
    from pops.relations import Accuracy, Algorithm, Approximation, Check, Evidence, Problem, RelationErrors, solve
    from pops.time.variant_types import Result

    def closed(inputs, output, body):
        return graph(inputs=Product(**{name: type_of(value) for name, value in inputs.items()}),
                     output=type_of(output))(body)

    def ok(g, value):
        return g.variant(Result(type_of(value), RelationErrors), "ok", value)

    def evidence(g):
        return Evidence(g, estimates=g.product(), valid=g.boolean(True))

    def approximation(g, values):
        return Approximation(g, values=values, evidence=evidence(g), termination=g.product())

    unknown = p.product(q=measured)
    data = p.product(target=measured)
    parameters = p.product(atol=p.float64(1.e-12), rtol=p.float64(0.),
                           scale=p.float64(1.), request=p.product())
    budget = p.product(steps=p.int64(1))
    residual_sample = p.product(r=measured)
    args = dict(unknown=unknown, data=data)
    residual = closed(args, residual_sample, lambda g, a: g.product(
        r=apply(g, a["unknown"]["q"]) - a["data"]["target"]))
    predicate = closed(args, p.boolean(True), lambda g, a: g.boolean(True))
    check = ok(p, Check(p, accepted=p.boolean(True), observations=p.product()))
    request = closed(dict(args, residual=residual_sample, accuracy=parameters, evidence=evidence(p)),
        check, lambda g, a: ok(g, Check(g, accepted=g.boolean(True), observations=g.product())))
    norm = closed(dict(residual=residual_sample), measured,
        lambda g, a: g.branch(a["residual"]["r"] >= 0,
                             lambda b: a["residual"]["r"], lambda b: -a["residual"]["r"]))
    algorithm = closed(dict(data=data, guess=unknown, accuracy=parameters, budget=budget),
        ok(p, approximation(p, unknown)),
        lambda g, a: ok(g, approximation(g, g.product(q=a["data"]["target"]))))
    checked = solve(p, Problem(residual, predicate, predicate, predicate, request), data=data,
                    algorithm=Algorithm(algorithm), guess=unknown,
                    accuracy=Accuracy(parameters, norm), budget=budget)
    return p.require_success(checked)["values"]["q"]


@pytest.mark.parametrize("kind", ("branch_fold", "map_fold", "relation_residual"))
def test_nested_strict_scientific_contracts_survive_real_compiler_detachment(kind):
    from pops.codegen._orchestration_compile import build_program_model_graph
    from pops.codegen.program_models import prepare_program_authority
    from pops.codegen._shared_interface_evidence import _issue_shared_interface_codegen_evidence
    from pops.codegen.program_graph_lowering import _emit_resolved_program_graph
    from pops.codegen.program_balance_due import validate_balance_due_contract
    from pops._balance_due_contract import BalanceDueContract
    from pops.time._program.detach import detach_compiled_program
    from pops.time._program.temporal_manifest import _walk
    from pops.scientific_graph import CONTRACT, validate_scientific_program
    from pops.codegen.program_source_map import source_map
    plan = _nested_scientific_plan(kind)
    model_graph = build_program_model_graph(plan)
    prepare_program_authority(None, model_graph)
    contract = BalanceDueContract.from_consumer_graph(plan.consumer_graph)
    validate_balance_due_contract(plan.time, contract)
    strict_nodes = [v for v in _walk(plan.time._values)
                    if v.attrs.get(CONTRACT, {}).get("mode") == "strict"]
    assert strict_nodes
    assert all(all(v is not root for root in plan.time._values) for v in strict_nodes)
    before = plan.time._serialize(include_provenance=False)
    detached = detach_compiled_program(plan.time,
        physical_global_sources=plan._physical_global_sources)
    assert detached._serialize(include_provenance=False) == before
    assert validate_scientific_program(detached, strict=True)
    assert {identifier for identifier, _ in detached._scientific_compile_binding.proof.contracts} == {
        v.id for v in strict_nodes}
    cpp = _emit_resolved_program_graph(detached.to_graph(), lowering_program=detached,
        model=None, model_graph=model_graph, target=plan.target, field_plans=plan.field_plans,
        balance_due_contract=contract,
        shared_interface_codegen_evidence=_issue_shared_interface_codegen_evidence(plan))
    assert "ctx.commit_many(" in cpp and "pops_program_abi_key" in cpp
    assert source_map(cpp)["complete"]
    if kind == "relation_residual":
        from pops.relations import SolveFailure
        assert str(int(SolveFailure.ORIGINAL_RESIDUAL)) in cpp


# Exact-SSA alias specialization regressions.
"""Exact-SSA alias specialization of strict scientific applications; source only."""
import copy
import json

import pytest

from pops._ir.quantity import PhysicalDimension, PhysicalSupport
from pops.model.spaces import Space
from pops.representations import ScientificMap, ScientificType, ScientificValue
from pops.scientific_graph import (CONTRACT, APPLICATION, bind_scientific_value,
                                   validate_scientific_program, _key)
from pops.graph_functions import graph, GraphFunction
from pops.time import Program
from pops.time.variant_types import Float64, Product, type_of

ONE = PhysicalDimension()
LENGTH = PhysicalDimension((("length", 1),))
TIME = PhysicalDimension((("time", 1),))


def scientific(unit=ONE, *, finite=Float64, dofs=None, domain="real"):
    space = Space("q", ("q",), support=PhysicalSupport(()), units=(unit,),
                  value_shape=(), sampling="point", representation="coordinates", domain=domain)
    return ScientificType(finite, space, "point_value", (), (), {"name": "counting"}, dofs=dofs)


def mapping(typ=None, body=None):
    typ = typ or scientific()
    body = body or (lambda p, a: 2 * a["a"] - 3 * a["b"])
    fn = graph(inputs=Product(a=typ.finite_type, b=typ.finite_type),
               output=typ.finite_type)(body)
    return ScientificMap(fn, inputs={"a": typ, "b": typ}, output=typ, strict=True)


@pytest.mark.parametrize("same", (False, True))
def test_exact_alias_and_equal_but_distinct_values_are_accepted(same):
    p = Program("aliased")
    a = ScientificValue(p.float64(2), scientific())
    b = a if same else ScientificValue(p.float64(2), scientific())
    call = mapping()
    out = call(p, a=a, b=b)
    assert (a.value.id == b.value.id) is same
    assert validate_scientific_program(p, strict=True)
    assert _key(p._canonical_value(out.value).attrs[CONTRACT]) == _key(call.checked_contract.to_data())
    references = p._canonical_value(out.value).attrs[APPLICATION]["input_values"]
    assert references["a"].id == a.value.id and references["b"].id == b.value.id


@pytest.mark.parametrize("corruption", ("repeat_a", "swap", "constant"))
def test_equal_numbers_do_not_authorize_a_different_wiring(corruption):
    p = Program("bad_wiring")
    a = ScientificValue(p.float64(2), scientific()).value
    b = ScientificValue(p.float64(2), scientific()).value
    wrong = {"repeat_a": lambda: 2*a-3*a,
             "swap": lambda: 2*b-3*a,
             "constant": lambda: p.float64(-2)}[corruption]()
    with pytest.raises(TypeError, match="exact applied graph body"):
        bind_scientific_value(wrong, scientific(), contract=mapping().checked_contract,
                              bindings={"a": a, "b": b})


def test_false_alias_partition_cannot_hide_a_distinct_dependency():
    p = Program("false_partition")
    a = ScientificValue(p.float64(2), scientific()).value
    b = ScientificValue(p.float64(2), scientific()).value
    value = 2*a-3*b
    with pytest.raises(TypeError, match="exact applied graph body"):
        bind_scientific_value(value, scientific(), contract=mapping().checked_contract,
                              bindings={"a": a, "b": a})


def test_unused_named_alias_is_legal_but_all_declared_names_are_required():
    p = Program("unused")
    a = ScientificValue(p.float64(2), scientific())
    call = mapping(body=lambda g, x: 2*x["a"])
    assert validate_scientific_program(call(p, a=a, b=a).value.prog, strict=True)
    with pytest.raises(TypeError, match="exact named inputs"):
        bind_scientific_value(2*a.value, scientific(), contract=call.checked_contract,
                              bindings={"a": a.value})


@pytest.mark.parametrize("change", ("units", "domain", "owner"))
def test_aliasing_does_not_relax_scientific_types_or_owner(change):
    p = Program("type_owner")
    call = mapping(scientific(LENGTH))
    a = ScientificValue(p.float64(2), scientific(LENGTH))
    if change == "units":
        b = ScientificValue(p.float64(2), scientific(TIME))
    elif change == "domain":
        b = ScientificValue(p.float64(2), scientific(LENGTH, domain="positive"))
    else:
        other = Program("type_owner")
        b = ScientificValue(other.float64(2), scientific(LENGTH))
    before = p._serialize()
    with pytest.raises((TypeError, ValueError)):
        call(p, a=a, b=b)
    assert p._serialize() == before


@pytest.mark.parametrize("reload", (False, True))
def test_outer_substitution_can_create_aliases_in_a_stored_inner_contract(reload):
    call = mapping()
    @graph(inputs=Product(left=Float64, right=Float64), output=Float64)
    def outer(g, a):
        return g.branch(a["left"] >= 0,
            lambda b: b.fold(a["left"], 2, lambda c, state, index:
                call(c, a=ScientificValue(a["left"], scientific()),
                     b=ScientificValue(a["right"], scientific())).value),
            lambda b: call(b, a=ScientificValue(a["left"], scientific()),
                          b=ScientificValue(a["right"], scientific())).value)
    if reload:
        outer = GraphFunction.from_data(json.loads(json.dumps(outer.to_data())))
    p = Program("nested_alias")
    q = ScientificValue(p.float64(2), scientific())
    out = outer(p, left=q.value, right=q.value)
    assert out.prog is p and validate_scientific_program(p, strict=True)


def test_original_contract_is_rechecked_even_when_application_inputs_alias():
    p = Program("tamper")
    q = ScientificValue(p.float64(2), scientific())
    out = mapping()(p, a=q, b=q).value
    actual = p._canonical_value(out)
    bad = json.loads(_key(actual.attrs[CONTRACT]))
    bad["body_hash"] = "0" * 64
    p._replace_value(actual, attrs=dict(actual.attrs, **{CONTRACT: bad}))
    with pytest.raises(TypeError, match="metadata differs from rechecked graph"):
        validate_scientific_program(p, strict=True)


def _prepared_plan(mode):
    import pops
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.initial import InitialCondition
    from pops.layouts import Uniform
    from pops.lib.initial import BindArray
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, StateStorage
    from pops.projection import ConservativeCellAverage
    from pops.time import FixedDt
    from pops.time.expressions import ProgramScalar
    frame = Rectangle("alias_grid", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("alias_model", frame=frame)
    state = model.state("q", components=("q",), units=(ONE,))
    case = pops.Case("alias_case"); block = case.block("samples", model)
    numerics = DiscretizationPlan(); numerics.states.add(state, StateStorage())
    case.numerics(numerics, block=block)
    p = Program("alias_program"); q = p.state(block[state]); measured = p.norm_inf(q.n)
    measured = ScientificValue(measured, scientific(finite=type_of(measured))).value
    if mode == "array":
        domain = p.range_set(3, capacity=3, ownership="replicated")
        value = p.map(domain, lambda g, index: measured)
        typ = scientific(finite=type_of(value), dofs=type_of(domain))
        call = mapping(typ, lambda g, a: g.map(g.domain(a["a"]),
            lambda z, index: 2*z.at(a["a"], index)-3*z.at(a["b"], index)))
        a = ScientificValue(value, typ)
        array = call(p, a=a, b=a).value
        result = p.reduce(array, measured*0, lambda g, total, x: total+x, order="ascending")
    else:
        typ = scientific(finite=type_of(measured)); a = ScientificValue(measured, typ)
        call = mapping(typ)
        result = p.branch(measured >= 0,
            lambda b: b.fold(measured, 2, lambda g, carry, index: call(g, a=a, b=a).value),
            lambda b: call(b, a=a, b=a).value)
    p.commit(q.next, p.value("accepted", (q.n[0]+ProgramScalar(result),), at=q.next.point))
    p.step_strategy(FixedDt(.01)); case.program(p)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    return pops.resolve(pops.validate(case), layout=Uniform(CartesianGrid(
        frame=frame, cells=(8, 4), periodic=PeriodicAxes(frame.axes))))


@pytest.mark.parametrize("mode", ("scalar_regions", "array"))
def test_aliases_survive_prepare_detach_redetach_and_real_cpp_emission(mode):
    from pops.codegen._orchestration_compile import build_program_model_graph
    from pops.codegen.program_models import prepare_program_authority
    from pops.codegen._shared_interface_evidence import _issue_shared_interface_codegen_evidence
    from pops.codegen.program_graph_lowering import _emit_resolved_program_graph
    from pops.codegen.program_balance_due import validate_balance_due_contract
    from pops._balance_due_contract import BalanceDueContract
    from pops.time._program.detach import detach_compiled_program
    from pops.codegen.program_source_map import source_map
    plan = _prepared_plan(mode)
    models = build_program_model_graph(plan); prepare_program_authority(None, models)
    balance = BalanceDueContract.from_consumer_graph(plan.consumer_graph)
    validate_balance_due_contract(plan.time, balance)
    image = plan.time._serialize(include_provenance=False)
    detached = detach_compiled_program(plan.time, physical_global_sources=plan._physical_global_sources)
    assert detached._serialize(include_provenance=False) == image
    assert validate_scientific_program(detached, strict=True)
    assert validate_scientific_program(detach_compiled_program(detached), strict=True)
    cpp = _emit_resolved_program_graph(detached.to_graph(), lowering_program=detached,
        model=None, model_graph=models, target=plan.target, field_plans=plan.field_plans,
        balance_due_contract=balance,
        shared_interface_codegen_evidence=_issue_shared_interface_codegen_evidence(plan))
    assert "ctx.commit_many(" in cpp and source_map(cpp)["complete"]


@pytest.mark.parametrize("reverse", (False, True))
def test_partial_alias_partition_with_permuted_keywords_and_unused_argument(reverse):
    typ = scientific()
    fn = graph(inputs=Product(c=Float64, a=Float64, b=Float64, unused=Float64),
               output=Float64)(lambda g, x: x["a"]-2*x["b"]+3*x["c"])
    call = ScientificMap(fn, inputs={name: typ for name in ("c", "a", "b", "unused")},
                         output=typ, strict=True)
    p = Program("partial_alias")
    a = ScientificValue(p.float64(2), typ)
    b = ScientificValue(p.float64(5), typ)
    unused = ScientificValue(p.float64(9), typ)
    bindings = dict(a=a, b=b, c=a, unused=unused)
    if reverse:
        bindings = dict(reversed(tuple(bindings.items())))
    out = call(p, **bindings).value
    assert validate_scientific_program(p, strict=True)
    # Rebinding order cannot change the specialized body identity.
    node = p._canonical_value(out)
    application = dict(node.attrs[APPLICATION])
    application["input_values"] = dict(reversed(tuple(application["input_values"].items())))
    p._replace_value(node, attrs=dict(node.attrs, **{APPLICATION: application}))
    assert validate_scientific_program(p, strict=True)
