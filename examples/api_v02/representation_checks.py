"""Finite representation equations consumed by genuine native PoPS fields.

This is an algebra reception, not a claimed PDE model. The scalar map inputs
include a global native field reduction; NumPy appears only in initial data and
the independent post-execution oracle.
"""
from __future__ import annotations

import numpy as np
import pops

from pops._ir.quantity import PhysicalDimension, PhysicalSupport
from pops.claims import Claim
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.graph_functions import graph
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model.spaces import Space
from pops.numerics import DiscretizationPlan, StateStorage
from pops.projection import ConservativeCellAverage
from pops.representations import Representation, ScientificType, ScientificMap, ScientificValue
from pops.time import FixedDt
from pops.time.expressions import ProgramScalar
from pops.time.variant_types import Float64, Product, type_of

from .common import CaseSpec, non_owner_metrics, state_array


def finite_representation(*, encode_scale=2., decode_scale=3., pairing_scale=5.,
                           finite_type=Float64, dual_type=Float64):
    """Three independent Python equations; encode/decode need not be inverses."""
    constants = (encode_scale, decode_scale, pairing_scale)
    if any(type(v) not in (int, float) or not np.isfinite(v) for v in constants) or decode_scale == 0:
        raise ValueError("representation scales must be finite, with nonzero decoder")
    physical = Space("scalar_observable", ("q",), value_shape=(),
                     support=PhysicalSupport(()), units=(PhysicalDimension(),),
                     sampling="point", representation="physical_quantity")
    def scientific(kind, value_type=finite_type):
        return ScientificType(value_type, physical, kind, (), (), {"name": "counting"})
    sample, coefficient, pairing_value = map(scientific,
        ("point_value", "coefficient", "pairing_value"))
    dual = scientific("dual_charge", dual_type)

    @graph(inputs=Product(value=finite_type), output=finite_type)
    def encoder(p, args):
        return p.value("encoded", encode_scale*args["value"])

    @graph(inputs=Product(value=finite_type), output=finite_type)
    def decoder(p, args):
        return p.value("decoded", args["value"]/decode_scale)

    @graph(inputs=Product(left=finite_type, right=dual_type), output=finite_type)
    def pair(p, args):
        return p.value("paired", pairing_scale*args["left"]*args["right"])

    from pops.scientific_graph import ScientificConversion
    def conversion(function, purpose):
        return ScientificConversion(function.identity, purpose,
            'The authored finite equation defines this coordinate map; encode and decode are not asserted inverse.')
    encode = ScientificMap(encoder, inputs={"value": sample}, output=coefficient,
        strict=True, conversion=conversion(encoder, 'representation'))
    decode = ScientificMap(decoder, inputs={"value": coefficient}, output=sample,
        strict=True, conversion=conversion(decoder, 'representation'))
    pairing = ScientificMap(pair, inputs={"left": coefficient, "right": dual}, output=pairing_value,
        strict=True, conversion=conversion(pair, 'pairing'),
        claims=(Claim.unestablished("physical adjoint interpretation",
                                   reason="this finite algebra check asserts no continuum adjoint"),))
    return Representation(physical, dofs=None, kind="coefficient", encode=encode,
                          decode=decode, pairing=pairing)


def build_representation_check(*, encode_scale=2., decode_scale=3., pairing_scale=5.):
    frame = Rectangle("representation_grid", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("finite_representation_equations", frame=frame)
    U = model.state("q", components=("q",), units=(PhysicalDimension(),))
    case = pops.Case("finite_representation_algebra")
    block = case.block("samples", model)
    plan = DiscretizationPlan()
    plan.states.add(U, StateStorage())
    case.numerics(plan, block=block)
    p = pops.Program("encode_decode_pairing")
    q = p.state(block[U])
    measured_value = p.norm_inf(q.n)
    dual_value = p.float64(1.25)
    rep = finite_representation(encode_scale=encode_scale, decode_scale=decode_scale,
                                pairing_scale=pairing_scale, finite_type=type_of(measured_value),
                                dual_type=type_of(dual_value))
    observable = ScientificValue(measured_value, rep.encode.inputs["value"])
    encoded = rep.encode(p, value=observable)
    decoded = rep.decode(p, value=encoded)
    dual = ScientificValue(dual_value, rep.pairing.inputs["right"])
    paired = rep.pairing(p, left=encoded, right=dual)
    # This algebra receiver sums coordinates of three different representations.
    # The coordinate maps are explicit; extracting .value is not a conversion.
    def coordinates(value):
        from pops.scientific_graph import ScientificConversion
        scalar = ScientificType(type_of(value.value),
            Space('receiver_coordinate', ('q',), value_shape=(), support=PhysicalSupport(()),
                  units=(PhysicalDimension(),), sampling='point', representation='coordinates'),
            'point_value', (), (), {'name': 'counting'})
        coordinate_map = graph(inputs=Product(value=type_of(value.value)),
            output=type_of(value.value))(lambda g, a: a['value'])
        return ScientificMap(coordinate_map, inputs={'value': value.scientific_type}, output=scalar,
            strict=True, conversion=ScientificConversion(coordinate_map.identity, 'representation',
                'This test reads a dimensionless scalar coordinate; it does not identify the represented quantities.'
            ))(p, value=value).value
    delta = coordinates(encoded) + coordinates(decoded) + coordinates(paired)
    p.commit(q.next, p.value("accepted", (q.n[0]+ProgramScalar(delta),), at=q.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    initial = (.2 + np.arange(32, dtype=float).reshape(1, 4, 8)/64)
    subject = block[U]
    case.initials.add(InitialCondition(state=subject, value=BindArray(), projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 4), periodic=PeriodicAxes(frame.axes)))
    # Independent arithmetic reference; never evaluate the GraphFunction in Python.
    measured = float(np.max(np.abs(initial)))
    expected_delta = encode_scale*measured*(1 + 1/decode_scale + pairing_scale*1.25)
    expected = initial + expected_delta

    def invariants(runtime):
        actual = state_array(runtime, "samples", initial.shape)
        if actual is None:
            return non_owner_metrics()
        error = float(np.max(np.abs(actual-expected)))
        return {"metrics_owner": True, "max_error": error, "expected_delta": expected_delta,
                "inverse_assumption_gap": abs(encode_scale/decode_scale-1)*measured,
                "passed": bool(error < 2e-13 and np.all(np.isfinite(actual)))}

    return CaseSpec(case, layout, {subject: initial}, {"samples": subject}, .01, .01,
                    invariants, "typed encode/decode/pairing on global native observable, CPU Uniform Dim2")


def build_scientific_alias_check(*, aliased):
    """Receive a strict noncommutative map on exact shared or distinct native SSA."""
    if type(aliased) is not bool:
        raise TypeError("aliased must be an explicit Bool")
    frame = Rectangle("alias_grid", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("scientific_alias_equations", frame=frame)
    U = model.state("q", components=("q",), units=(PhysicalDimension(),))
    case = pops.Case("scientific_alias_algebra")
    block = case.block("samples", model)
    plan = DiscretizationPlan()
    plan.states.add(U, StateStorage())
    case.numerics(plan, block=block)
    p = pops.Program("strict_two_argument_map")
    q = p.state(block[U])
    left = p.norm_inf(q.n)
    if aliased:
        right = left  # The very same ProgramValue; no synthetic de-aliasing operation.
    else:
        doubled = p.value("doubled_input", (2*q.n[0],), at=q.n.point)
        right = p.norm_inf(doubled)
    assert (left is right) is aliased
    physical = Space("alias_observable", ("q",), value_shape=(),
                     support=PhysicalSupport(()), units=(PhysicalDimension(),),
                     sampling="point", representation="physical_quantity")
    def scientific(value):
        return ScientificType(type_of(value), physical, "point_value", (), (), {"name": "counting"})
    left_type, right_type = scientific(left), scientific(right)

    @graph(inputs=Product(left=type_of(left), right=type_of(right)), output=type_of(left))
    def combine(g, args):
        return g.value("noncommutative_delta", 2*args["left"] - 3*args["right"])

    mapping = ScientificMap(combine, inputs={"left": left_type, "right": right_type},
                            output=left_type, strict=True)
    delta = mapping(p, left=ScientificValue(left, left_type),
                    right=ScientificValue(right, right_type)).value
    p.commit(q.next, p.value("accepted", (q.n[0]+ProgramScalar(delta),), at=q.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    initial = .2 + np.arange(32, dtype=float).reshape(1, 4, 8)/64
    subject = block[U]
    case.initials.add(InitialCondition(state=subject, value=BindArray(), projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 4), periodic=PeriodicAxes(frame.axes)))
    # Independent arithmetic on initial data only; no GraphFunction evaluation.
    first = float(np.max(np.abs(initial)))
    second = first if aliased else float(np.max(np.abs(2*initial)))
    expected_delta = 2*first - 3*second
    expected = initial + expected_delta
    swapped_delta = 2*second - 3*first

    def invariants(runtime):
        actual = state_array(runtime, "samples", initial.shape)
        if actual is None:
            return non_owner_metrics()
        error = float(np.max(np.abs(actual-expected)))
        return {"metrics_owner": True, "max_error": error,
                "arguments_aliased": aliased, "expected_delta": expected_delta,
                "stationary_counterexample_error": abs(expected_delta),
                "swapped_counterexample_gap": abs(swapped_delta-expected_delta),
                "graph_identity": combine.identity,
                "passed": bool(error < 2e-13 and np.all(np.isfinite(actual)))}

    return CaseSpec(case, layout, {subject: initial}, {"samples": subject}, .01, .01,
                    invariants, "strict shared/distinct SSA map on native field reductions, CPU Uniform Dim2")


__all__ = ["finite_representation", "build_representation_check", "build_scientific_alias_check"]
