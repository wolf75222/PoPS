"""Contract @2 of a representation (spec 0.3 D1, D1b, D2, D4; G03 lot R; N16 rev 3, 5.1, 5.2, 5.5).

Source-only: the definition, its two identities, the derived data of one constant mode and of a
polynomial decode, the obligations, and ``MeasureMissing`` (spec 14.1 as amended by delta 5).
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from pops.model.spaces import StateSpace
from pops.representations.definition import (
    CONTRACT, LATER_LOT, NO_EXACT_FORM, NO_MEASURE, OBLIGATIONS, ElaboratedRepresentation,
    MeasureMissing, RepresentationDefinition, Unknown, body_from_data, call, cell, cell_average,
    cell_total, cells_not_affine, coefficients, constant_mode_scale, field, inventory_weights,
    node_reference, normal_form, point, point_value,
)
from pops.time._program.traced_method import NotYetElaborated

MEASURE = (("cell.measure", ()), 1)
INVERSE_MEASURE = (("cell.measure", ()), -1)


class Rule:
    """A rule as a case writes it: any value with nodes, weights and interval (D0)."""

    def __init__(self, nodes, weights, interval=(-1, 1)):
        self.nodes, self.weights, self.interval = nodes, weights, interval


GL2 = Rule((-0.5773502691896257, 0.5773502691896257), (1.0, 1.0))
TRAPEZOID = Rule((-1, 1), (1, 1))


def test_contract_names_and_obligations():
    assert CONTRACT == "pops.executable-representation-definition@2"
    data = cell_average(1).to_data()
    assert data["contract"] == CONTRACT
    assert data["obligations"] == list(OBLIGATIONS)
    assert "inventory_from_rate_weights" in OBLIGATIONS
    assert "encode_is_not_inverse_of_decode" in OBLIGATIONS


# --- one constant mode (N16 5.1) -----------------------------------------------------------------

def test_cell_average_has_c_one_and_today_weights():
    average = cell_average(2)
    assert average.constant_mode == {(): Fraction(1)}
    assert average.rate_weights() == ({(MEASURE,): Fraction(1)},)          # m_K = mu(K)
    assert average.load_weights() == ({(): Fraction(1)},)                  # w_K = 1
    assert average.riesz_scale() == {(INVERSE_MEASURE,): Fraction(1)}      # 1 / mu(K)
    assert constant_mode_scale(average) == "cell_measure"


def test_cell_total_has_c_inverse_measure_and_no_division():
    total = cell_total(2)
    assert total.constant_mode == {(INVERSE_MEASURE,): Fraction(1)}
    assert total.rate_weights() == ({(): Fraction(1)},)                    # m_K = 1
    assert total.load_weights() == ({(MEASURE,): Fraction(1)},)            # w_K = mu(K)
    assert total.riesz_scale() == {(): Fraction(1)}                         # no division
    assert constant_mode_scale(total) == "unit"


def test_point_value_and_stated_rule_lower_as_one_mode_with_c_one():
    assert constant_mode_scale(point_value(1)) == "cell_measure"
    assert constant_mode_scale(cell_average(1, quadrature=GL2)) == "cell_measure"
    # the rule does not enter the weights of one constant mode (N16 3.3, A3 of N16)
    assert cell_average(1, quadrature=GL2).rate_weights() == cell_average(1).rate_weights()


def test_the_factor_may_be_written_any_way_that_normalizes():
    spelled = RepresentationDefinition(decode=2 * coefficients[0] / (2 * cell.measure), rank=1,
                                       kind="total")
    assert spelled.constant_mode == cell_total(1).constant_mode
    assert constant_mode_scale(spelled) == "unit"
    # a point-free factor of width reads is a constant mode, but not one lot R lowers
    scaled = RepresentationDefinition(decode=coefficients[0] * cell.width(0), rank=1)
    assert scaled.constant_mode == {((("cell.width", (0,)), 1),): Fraction(1)}
    assert constant_mode_scale(scaled) is None
    assert not isinstance(scaled.riesz_scale(), Unknown)   # 1 / (h mu) is a monomial


def test_point_dependent_decode_is_no_constant_mode():
    modal = RepresentationDefinition(decode=coefficients[0] + coefficients[1] * point.reference[0],
                                     rank=1, size=2, pairing=GL2)
    assert modal.point_dependent is True
    assert modal.linear is True
    assert isinstance(modal.constant_mode, Unknown)
    assert isinstance(modal.riesz_scale(), Unknown)
    assert constant_mode_scale(modal) is None
    # coefficients[0] + 0 * point.reference does not depend on the point (N16 3.2)
    flat = RepresentationDefinition(decode=coefficients[0] + 0 * point.reference[0], rank=1)
    assert flat.point_dependent is False
    average = cell_average(1)
    # the zero term vanishes from the normal form: the decode is the one of cell_average ...
    assert flat.normalized_data()["decode"] == average.normalized_data()["decode"]
    # ... but the kind is part of the identity (O2 of the G03 review): coefficient is not average
    assert flat.identity != average.identity
    same_kind = RepresentationDefinition(decode=coefficients[0] + 0 * point.reference[0], rank=1,
                                         kind="average")
    assert same_kind.identity == average.identity


# --- general rate and load weights (N16 5.2, A3 of the G03 review) --------------------------------

def test_modal_p1_weights_exact():
    modal = RepresentationDefinition(decode=coefficients[0] + coefficients[1] * point.reference[0],
                                     rank=1, size=2)
    m = modal.rate_weights()
    assert m == ({(MEASURE,): Fraction(1)}, {})                            # int xi = 0
    assert modal.load_weights() == ({(): Fraction(1)}, {})


def test_lagrange_p1_weights_exact():
    xi = point.reference[0]
    hat = RepresentationDefinition(decode=coefficients[0] * (1 - xi) / 2
                                   + coefficients[1] * (1 + xi) / 2, rank=1, size=2)
    assert hat.rate_weights() == ({(MEASURE,): Fraction(1, 2)}, {(MEASURE,): Fraction(1, 2)})
    assert hat.load_weights() == ({(): Fraction(1)}, {(): Fraction(1)})


def test_rank_two_tensor_moments():
    x, y = point.reference[0], point.reference[1]
    q1 = RepresentationDefinition(decode=coefficients[0] + coefficients[1] * x * x
                                  + coefficients[2] * x * y, rank=2, size=3)
    m = q1.rate_weights()
    assert m == ({(MEASURE,): Fraction(1)}, {(MEASURE,): Fraction(1, 3)}, {})


def test_one_not_in_span_keeps_rate_weights_and_loses_load_weights():
    odd = RepresentationDefinition(decode=coefficients[0] * point.reference[0], rank=1)
    assert odd.rate_weights() == ({},)                     # m_K exists: the measure exists
    weights = odd.load_weights()
    assert isinstance(weights, Unknown) and "1 is not in the span" in weights.reason
    # the inventory is formed: MeasureMissing is not raised (delta 5); a conservation claim on
    # the balance is ConservationNotStructural's (IR17), not this lot's
    assert inventory_weights("u", odd) == ({},)


def test_measure_missing_when_no_measure():
    nomeasure = RepresentationDefinition(decode=coefficients[0], rank=1, measure="none")
    assert isinstance(nomeasure.rate_weights(), Unknown)
    with pytest.raises(MeasureMissing) as refused:
        inventory_weights("u", nomeasure)
    assert "MeasureMissing (D4, validate)" in str(refused.value)
    assert "no measure" in str(refused.value)
    assert refused.value.representation == nomeasure.identity


def test_measure_missing_when_not_normalizable_and_no_rule():
    opaque = RepresentationDefinition(decode=coefficients[0] * call("exp", point.reference[0]),
                                      rank=1)
    with pytest.raises(MeasureMissing):
        inventory_weights("u", opaque)
    ruled = RepresentationDefinition(decode=coefficients[0] * call("exp", point.reference[0]),
                                     rank=1, pairing=GL2)
    reason = ruled.rate_weights()
    assert isinstance(reason, Unknown) and "N16 lot 2" in reason.reason
    assert (reason.family, reason.lot) == (LATER_LOT, reason.lot) and "N16 lot 2" in reason.lot


def test_only_the_two_triggers_of_delta_5_are_measure_missing():
    """F2 of the review of #789: a weight that a later lot gives is NotYetElaborated, naming it."""
    xi = point.reference[0]
    nomeasure = RepresentationDefinition(decode=coefficients[0], rank=1, measure="none")
    assert nomeasure.rate_weights().family == NO_MEASURE
    later = {
        "a stated rule over an opaque decode": RepresentationDefinition(
            decode=coefficients[0] * call("exp", xi), rank=1, pairing=GL2),
        "a stated rule over the position": RepresentationDefinition(
            decode=coefficients[0] * point.position[0], rank=1, pairing=GL2),
        "a stated rule over a decode not linear in the coefficients": RepresentationDefinition(
            decode=coefficients[0] * coefficients[0], rank=1, pairing=GL2),
        "a face carrier": RepresentationDefinition(
            decode=coefficients[0], rank=1, carrier="face", encode=field[0]),
    }
    for case, definition in later.items():
        weights = definition.rate_weights()
        assert isinstance(weights, Unknown) and weights.family == LATER_LOT, case
        with pytest.raises(NotYetElaborated, match="N16 lot 2|G08") as refused:
            inventory_weights("u", definition)
        assert not isinstance(refused.value, MeasureMissing), case
    # normalizable but not linear in the coefficients, no rule: no basis, so MeasureMissing, with
    # the reason the delta reads as « not normalizable »
    square = RepresentationDefinition(decode=coefficients[0] * coefficients[0], rank=1)
    with pytest.raises(MeasureMissing, match="not linear in the coefficients, and the "
                                             "representation states no rule"):
        inventory_weights("u", square)


def _n17_layout(builder):
    import pops
    from pops.mesh import normalize_layout_plan
    _, layout, _, _ = builder()
    plan = normalize_layout_plan(layout, owner=pops.Case("g03-ra").owner_path.canonical())
    (row,) = plan.layouts
    return row


def test_moments_of_the_reference_box_are_refused_on_a_chart():
    """Review of #789, O1 made blocking: m_K of a decode that reads the point is a moment of the
    reference box, exact on affine cells only. On N17's annulus chart it is refused by name; on
    N17's graded node positions (affine cells) and for one constant mode it holds."""
    from examples.api_v02.models_static_geometry import (
        build_annulus_free_stream, build_graded_diffusion)
    annulus, graded = _n17_layout(build_annulus_free_stream), _n17_layout(build_graded_diffusion)
    assert "chart" in cells_not_affine(annulus) and cells_not_affine(graded) is None
    assert cells_not_affine(None) is None
    modal = RepresentationDefinition(decode=coefficients[0] + coefficients[1] * point.reference[0],
                                     rank=1, size=2, pairing=GL2)
    with pytest.raises(NotYetElaborated, match="affine cells only.*chart.*N16 lot 2"):
        inventory_weights("u", modal, layout=annulus)
    assert inventory_weights("u", modal, layout=graded) == modal.rate_weights()
    for one_mode in (cell_average(2), cell_total(2), point_value(2)):
        assert inventory_weights("u", one_mode, layout=annulus) == one_mode.rate_weights()


def test_a_moving_map_is_not_affine():
    from tests.python.unit.mesh.test_moving_geometry_contract import request
    import pops
    from pops.mesh import normalize_layout_plan
    _, moving = request()
    plan = normalize_layout_plan(moving, owner=pops.Case("g03-ra-moving").owner_path.canonical())
    (row,) = plan.layouts
    assert "G03 lots M1 and M2" in cells_not_affine(row)


# --- identities (N16 5.5) -------------------------------------------------------------------------

def test_definition_identity_is_the_normal_form_not_the_spelling():
    xi = point.reference[0]
    modal = RepresentationDefinition(decode=coefficients[0] + coefficients[1] * xi, rank=1,
                                     size=2, pairing=GL2)
    legendre = RepresentationDefinition(decode=coefficients[1] * xi + 1 * coefficients[0], rank=1,
                                        size=2, pairing=GL2)
    assert modal.identity == legendre.identity
    assert modal == legendre and hash(modal) == hash(legendre)


def test_definition_identity_separates_what_differs():
    base = cell_average(1)
    assert base.identity != point_value(1).identity            # encode differs
    assert base.identity != cell_total(1).identity             # decode differs
    assert base.identity != cell_average(1, quadrature=GL2).identity      # rule differs
    assert cell_average(1, quadrature=GL2).identity != cell_average(1, quadrature=TRAPEZOID).identity
    assert base.identity != cell_average(2).identity           # rank differs
    relabelled = RepresentationDefinition(decode=coefficients[0], rank=1, kind="coefficient")
    assert base.identity != relabelled.identity                # kind is in the digest (O2)
    nomeasure = RepresentationDefinition(decode=coefficients[0], rank=1, kind="average",
                                         measure="none")
    assert base.identity != nomeasure.identity


def test_a_rule_one_ulp_apart_is_another_identity():
    nudged = Rule((-0.5773502691896257, 0.5773502691896258), (1.0, 1.0))
    assert cell_average(1, quadrature=GL2).identity != cell_average(1, quadrature=nudged).identity


def test_elaborated_identity_adds_the_space():
    rho = StateSpace("rho", components=("rho",))
    energy = StateSpace("E", components=("E",))
    a = ElaboratedRepresentation(rho, cell_average(1))
    assert a.identity == ElaboratedRepresentation(rho, cell_average(1)).identity
    assert a.identity != ElaboratedRepresentation(energy, cell_average(1)).identity
    assert a.identity != ElaboratedRepresentation(rho, cell_total(1)).identity
    assert a.to_data()["definition"] == cell_average(1).identity


def test_equal_shapes_are_not_interchangeable():
    rho = StateSpace("rho", components=("rho",))
    average = ElaboratedRepresentation(rho, cell_average(1))
    total = ElaboratedRepresentation(rho, cell_total(1))
    average.require_same(ElaboratedRepresentation(rho, cell_average(1)), where="read")
    with pytest.raises(TypeError, match="equal shapes do not make them interchangeable"):
        average.require_same(total, where="read of rho")
    with pytest.raises(TypeError):
        average.require_same("an array", where="read of rho")


# --- data and refusals ----------------------------------------------------------------------------

def test_bodies_round_trip_through_data():
    xi = point.reference[0]
    body = (coefficients[0] + coefficients[1] * xi) / cell.measure - call("sin", xi) ** 2
    again = body_from_data(RepresentationDefinition(decode=body, rank=1, size=2).to_data()
                           ["decode_body"])
    assert normal_form(again) == normal_form(body)


def test_encode_and_decode_read_only_their_entities():
    with pytest.raises(ValueError, match="never field"):
        RepresentationDefinition(decode=field[0], rank=1)
    with pytest.raises(ValueError, match="never coefficients or point"):
        RepresentationDefinition(decode=coefficients[0], rank=1, encode=point.reference[0])
    # inside a call whose argument has one, two or three terms (review of #789, F1)
    xi = point.reference[0]
    for argument in (field[0], field[0] + 1, field[0] + 1 + field[0] * field[0]):
        with pytest.raises(ValueError, match="never field"):
            RepresentationDefinition(decode=coefficients[0] * call("exp", argument), rank=1)
    for argument in (xi, xi + 1, xi + 1 + xi * xi):
        with pytest.raises(ValueError, match="never coefficients or point"):
            RepresentationDefinition(decode=coefficients[0], rank=1,
                                     encode=field[0] * call("exp", argument))
    with pytest.raises(ValueError, match="never coefficients or point"):
        RepresentationDefinition(decode=coefficients[0], rank=1,
                                 encode=field[0] / (xi + 1))
    with pytest.raises(ValueError, match="never field"):
        RepresentationDefinition(decode=coefficients[0] * (field[0] + 2) ** cell.measure, rank=1)


@pytest.mark.parametrize("body", (
    coefficients[0] * call("exp", point.reference[0]),
    coefficients[0] * call("exp", point.reference[0] + 1),
    coefficients[0] * call("atan2", point.reference[0] + 1, 1),
    coefficients[0] * call("exp", point.reference[0] + 1 + point.reference[0] ** 2),
    coefficients[0] / (point.reference[0] + 3),
    coefficients[0] * (point.reference[0] + 2) ** cell.measure,
))
def test_a_point_read_inside_an_opaque_atom_is_seen(body):
    """F1 of the review of #789: the arguments of a call are read one normal form at a time."""
    definition = RepresentationDefinition(decode=body, rank=1)
    assert definition.point_dependent is True
    assert isinstance(definition.constant_mode, Unknown)
    weights = definition.rate_weights()
    assert isinstance(weights, Unknown) and weights.family == NO_EXACT_FORM
    with pytest.raises(MeasureMissing, match="states no rule"):
        inventory_weights("u", definition)
    with pytest.raises(ValueError, match="encode is required"):
        RepresentationDefinition(decode=coefficients[0], rank=1, carrier="node")


def test_indices_inside_the_cell():
    with pytest.raises(ValueError, match="outside the local set"):
        RepresentationDefinition(decode=coefficients[1], rank=1)
    with pytest.raises(ValueError, match="outside a cell of rank 1"):
        RepresentationDefinition(decode=coefficients[0] * point.reference[1], rank=1)
    with pytest.raises(ValueError, match="outside a cell of rank 1"):
        RepresentationDefinition(decode=coefficients[0] * node_reference[2][0], rank=1)
    with pytest.raises(TypeError, match="Python integer"):
        coefficients[1.0]


def test_body_data_outside_the_closed_lists_is_refused():
    """Review of #789, O4: data cannot build a leaf, an operation or a constant that the leaves
    cannot."""
    for forged in ({"leaf": "cell.volume", "indices": []}, {"leaf": "coefficients", "indices": [-1]},
                   {"op": "max", "args": [{"const": [1, 1]}, {"const": [2, 1]}]},
                   {"op": "neg", "args": [{"const": [1, 1]}, {"const": [2, 1]}]},
                   {"const": [1, 0]}, {"call": "", "args": []}, ["leaf"]):
        with pytest.raises(ValueError):
            body_from_data(forged)


def test_the_normal_form_shares_the_algebra_of_the_surface():
    """Review of #789, O3: the core cannot import the surface's normal form (CC2), so this module
    has its own until N11's replaces both. Their products and sums of monomials agree on the same
    data, and the one deliberate difference is stated: a quotient by a monomial of geometry reads
    (measures and widths, positive by the plan's certificate) divides here, where the surface keeps
    every quotient by an expression an atom."""
    from pops.representations import definition as core
    from pops.surface import _scheme as surface
    a, b, c = (("leaf", "a"), 1), (("leaf", "b"), 1), (("leaf", "c"), 2)
    left = {(a,): Fraction(2), (b, c): Fraction(-1, 3), (): Fraction(5)}
    right = {(a,): Fraction(1, 2), (c,): Fraction(7)}
    assert core._times(left, right) == surface._times(left, right)
    for sign in (1, -1):
        assert core._plus(left, right, sign) == surface._plus(left, right, sign)
    mu, coefficient = cell.measure, coefficients[0]
    inverse = {normal_form(coefficient / mu), normal_form((1 / mu) * coefficient),
               normal_form(coefficient * mu ** -1)}
    direct = {normal_form(coefficient * mu), normal_form(mu * coefficient),
              normal_form((mu ** 2 / mu) * coefficient)}
    assert len(inverse) == 1 and len(direct) == 1 and inverse != direct


def test_a_float_is_its_exact_binary_fraction():
    """Review of #789, O4: 0.1 is the binary64 0.1, not 1/10, as a rule one ulp apart is another."""
    assert (RepresentationDefinition(decode=0.1 * coefficients[0], rank=1).identity
            != RepresentationDefinition(decode=Fraction(1, 10) * coefficients[0], rank=1).identity)


def test_closed_word_lists_and_rules():
    with pytest.raises(ValueError, match="carrier is one of"):
        RepresentationDefinition(decode=coefficients[0], rank=1, carrier="volume")
    with pytest.raises(ValueError, match="kind is one of"):
        RepresentationDefinition(decode=coefficients[0], rank=1, kind="density")
    with pytest.raises(ValueError, match="resolves it against the configuration"):
        RepresentationDefinition(decode=coefficients[0], rank=1, measure="lebesgue")
    with pytest.raises(TypeError, match="pairing is a pops.Quadrature"):
        RepresentationDefinition(decode=coefficients[0], rank=1, pairing=(1, 2))
    with pytest.raises(ValueError, match="as many finite weights"):
        RepresentationDefinition(decode=coefficients[0], rank=1, pairing=Rule((0,), (1, 1)))
    with pytest.raises(TypeError, match="no truth value"):
        bool(coefficients[0])


def test_surface_quadrature_is_accepted_as_a_rule():
    import pops.surface as surface
    rule = surface.Quadrature(nodes=(-0.5773502691896257, 0.5773502691896257),
                              weights=(1.0, 1.0), interval=(-1, 1))
    assert cell_average(1, quadrature=rule).identity == cell_average(1, quadrature=GL2).identity


def test_derived_data_is_inert_json():
    import json
    for definition in (cell_average(2), cell_total(2), point_value(1)):
        json.dumps(definition.derived_data())
        json.dumps(definition.to_data())
