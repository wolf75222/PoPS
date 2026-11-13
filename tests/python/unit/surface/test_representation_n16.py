"""N16 lot 1 (spec 0.3, D1, D1b, D2; N16 design note rev 3, sections 3, 4.4, 5.5 and 11; ROM-109):
``@pops.representation``, ``pops.Coefficients``, ``pops.Dual`` and the words of
``pops.lib.representations``.

- CF4: each cell word of the library has the definition identity of the same representation
  written in a case, and of the oracle of contract @2 (G03 lot R-a), at every rank; the identity
  digests normal forms, never a name (P10).
- A6: two representations whose normal forms differ have two identities (``PointValue()`` and
  ``CellAverage()`` share their decode and differ by their encode and kind).
- SU-T3: the L1 case of §17.1 under ``CellAverage()`` of the library, or under the same written in
  the case, gives the program IR, the artifact, the semantic identity and the generated C++ that it
  gives under ``pops.CellAverage()``: the transitional table of 5.5 matches the definition
  identity, and no result, identity or manifest changes.
- SU-T2: the refusals of 4.4 that lot 1 meets, each at the user's line, with the lot that
  elaborates what it refuses. The statement that a refusal names ends with a marker comment,
  ``# <- name``, as in ``test_su_t2_refusals.py``.

The oracle tests on a graded layout (5.1, A1) land with the lots that serve these words at the
surface: G03 lot M1 for ``CellTotal()``, and the N16 lot that retires the transitional table of
``CellAverage()`` (the projection by encode) for ``PointValue()`` and
``CellAverage(quadrature=...)``. Until then ``pops.Numerics`` refuses them by name, which this file
checks (review of #800, R1).
"""
import json
import os

import pytest

import pops
import pops.surface as surface
from pops.lib.numerics import two_point, upwind
from pops.lib.quadrature import GaussLegendre
from pops.lib.representations import (
    CellAverage, CellTotal, DualLoad, EdgeCirculation, FaceNormalIntegral, PointValue,
)
from pops.lib.time.methods import ssprk2
from pops.representations import definition as contract
from pops.surface.discrete import Cell, Edge, Face, Node
from pops.surface.math import ddt, div, grad

HERE = os.path.abspath(__file__)
RANKS = (1, 2, 3)


def _line(marker):
    lines = [number for number, text in enumerate(open(HERE, encoding="utf-8"), start=1)
             if text.rstrip().endswith("# <- " + marker)]
    assert len(lines) == 1, "marker %r is not on exactly one line" % marker
    return lines[0]


def _refused(caught, kind, marker, owner=None, write=None, traced=None):
    """A refusal of the surface of class ``kind`` at the marked line, naming ``owner`` (the lot
    that elaborates what it refuses) or the construct to ``write``, traced at ``traced``."""
    error = caught.value
    assert type(error) is kind
    line = _line(marker)
    assert (error.site.file, error.site.line) == (HERE, line)
    message = str(error)
    assert message.startswith("%s:%d: %s: " % (HERE, line, kind.__name__))
    if owner is not None:
        assert error.owner == owner or owner in message
    if write is not None:
        assert "Write %s" % write in message
    if traced is not None:
        assert "The case is traced at %s:%d." % (HERE, _line(traced)) in message
    return message


# --- the same representations, written in a case -------------------------------------------------

@surface.representation
def case_average(cell):
    def decode(coefficients, point):
        return coefficients[0]

    return surface.Coefficients(decode=decode, kind="average")


@surface.representation
def case_total(cell):
    def decode(coefficients, point):
        return coefficients[0] / cell.measure

    return surface.Coefficients(on=Cell, decode=decode, kind="total")


@surface.representation
def case_sample(cell):
    def decode(coefficients, point):
        return coefficients[0]

    def encode(cell, field):
        return [field]

    return surface.Coefficients(decode=decode, encode=encode, kind="sample")


@surface.representation
def case_average_with_rule(cell, rule=None):
    def decode(coefficients, point):
        return coefficients[0] + 0 * (point.reference if len(cell.axes) == 1
                                      else point.reference[0])

    return surface.Coefficients(decode=decode, pairing=rule, kind="average")


@surface.representation
def case_loads(cell):
    return surface.Dual(case_average())


# --- CF4: the library against the case and against contract @2 -----------------------------------

@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("word, written, oracle", [
    (CellAverage, case_average, contract.cell_average),
    (CellTotal, case_total, contract.cell_total),
    (PointValue, case_sample, contract.point_value),
])
def test_cf4_a_cell_word_has_the_identity_of_the_same_written_in_a_case(word, written, oracle,
                                                                         rank):
    library = word().definition(rank)
    assert isinstance(library, contract.RepresentationDefinition)
    assert library.identity == written().definition(rank).identity
    assert library.identity == oracle(rank).identity
    assert library.identity.startswith("pops.representation-definition:sha256:")
    assert word.name not in json.dumps(library.normalized_data())


@pytest.mark.parametrize("rank", RANKS)
def test_cf4_a_stated_rule_is_part_of_the_identity(rank):
    rule = GaussLegendre(3)
    stated = CellAverage(quadrature=rule).definition(rank)
    assert stated.identity == contract.cell_average(rank, rule).identity
    assert stated.identity == case_average_with_rule(rule=rule).definition(rank).identity
    assert stated.identity != CellAverage().definition(rank).identity
    # A3: the decode reads no point, so the rate weights are exact whatever rule is stated.
    assert stated.rate_weights() == CellAverage().definition(rank).rate_weights()


@pytest.mark.parametrize("rank", RANKS)
def test_cf4_dual_load_is_the_dual_of_the_cell_average(rank):
    loads = DualLoad().definition(rank)
    assert loads.kind == "load"
    assert loads.dual_of == CellAverage().definition(rank)
    assert loads.identity == case_loads().definition(rank).identity
    assert loads.identity == DualLoad(of=CellAverage()).definition(rank).identity
    assert DualLoad(of=CellTotal()).definition(rank).dual_of == CellTotal().definition(rank)
    assert loads.identity != CellAverage().definition(rank).identity


def test_pops_cell_average_elaborates_the_definition_of_the_library_word():
    for rank in RANKS:
        assert surface.CellAverage().definition(rank) == CellAverage().definition(rank)


# --- A6 and IR15: identities follow normal forms -------------------------------------------------

@pytest.mark.parametrize("rank", RANKS)
def test_a6_point_value_and_cell_average_have_two_identities(rank):
    identities = {word().definition(rank).identity for word in (CellAverage, CellTotal, PointValue)}
    assert len(identities) == 3
    assert not contract.RepresentationDefinition.__eq__(
        PointValue().definition(rank), CellAverage().definition(rank))


def test_ir15_point_dependence_is_derived_on_the_normal_form():
    # coefficients[0] + 0 * point.reference reads no point once normalized, so it is the cell
    # average itself, under any name.
    for rank in RANKS:
        written = case_average_with_rule().definition(rank)
        assert not written.point_dependent
        assert written.identity == CellAverage().definition(rank).identity


def test_a_representation_runs_once_per_rank_and_reads_its_cell():
    seen = []

    @surface.representation
    def counted(cell):
        seen.append((cell.rank, cell.axes, len(cell.nodes)))

        def decode(coefficients, point):
            return coefficients[0]

        return surface.Coefficients(decode=decode, kind="average")

    value = counted()
    assert seen == []                       # SU5: a call records the options and runs nothing
    value.definition(2)
    value.definition(2)
    value.definition(1)
    value.coefficients(2)                   # the body runs once per rank (review of #800, O1)
    assert seen == [(2, (0, 1), 4), (1, (0,), 2)]


def test_the_words_and_their_construct_are_exported():
    assert {"representation", "Coefficients", "Dual"} <= set(surface.__all__)
    assert {"Cell", "Node", "Face", "Edge"} <= set(surface.discrete.__all__)
    from pops.lib import representations
    assert representations.__all__ == ["CellAverage", "CellTotal", "DualLoad", "EdgeCirculation",
                                       "FaceNormalIntegral", "PointValue"]
    assert repr(CellAverage()) == "CellAverage()"
    assert repr(DualLoad(of=CellTotal())) == "DualLoad(of=CellTotal())"
    assert repr(CellAverage) == "@pops.representation CellAverage"


# --- SU-T3: the L1 case under each spelling of the cell average ----------------------------------

def _l1(representation):
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("advection_diffusion")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    nu = model.parameter("nu", default=1e-2, admissible=surface.Positive())
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=a * u)
    diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
    model.balance(ddt(u) == -div(advection + diffusion))
    numerics = surface.Numerics(represent={u: representation},
                                schemes={advection: upwind(), diffusion: two_point()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3, attempts=2),
                              initial={u: u0})
    return surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=64)})


def _forms(case):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph

    elaborated = case.elaborated
    resolved = pops.resolve(pops.validate(elaborated.case), layout=elaborated.layout)

    def without_provenance(value):
        if isinstance(value, dict):
            return {key: without_provenance(item) for key, item in value.items()
                    if key != "provenance"}
        if isinstance(value, list):
            return [without_provenance(item) for item in value]
        return value

    return {
        "program": elaborated.case._time._serialize(include_provenance=False),
        "artifact": without_provenance(json.loads(resolved.snapshot._artifact_canonical_json)),
        "semantic_identity": resolved.snapshot.semantic_identity,
        "cpp": emit_cpp_program(resolved.time, model_graph=ProgramModelGraph
                                .from_resolved_blocks(resolved.blocks)),
    }


@pytest.fixture(scope="module")
def b6_forms():
    return _forms(_l1(surface.CellAverage()))


@pytest.mark.parametrize("spelling", ["library", "case"])
def test_su_t3_the_cell_average_word_gives_the_forms_of_pops_cell_average(spelling, b6_forms):
    representation = CellAverage() if spelling == "library" else case_average()
    forms = _forms(_l1(representation))
    for name in ("program", "artifact", "semantic_identity", "cpp"):
        assert forms[name] == b6_forms[name], name


def test_su_t3_inspect_reports_the_representation_as_written():
    choices = [(choice.name, choice.value, choice.origin)
               for choice in surface.inspect(_l1(CellAverage())).choices()]
    assert ("representation", "CellAverage()", "stated") in choices
    choices = [(choice.name, choice.value, choice.origin)
               for choice in surface.inspect(_l1(surface.CellAverage())).choices()]
    assert ("representation", "pops.CellAverage()", "stated") in choices


# --- SU-T2: the refusals of 4.4 that lot 1 meets -------------------------------------------------

def _quantity():
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("m")
    return model.quantity("u", surface.Scalar(domain))


def test_a_total_is_refused_until_g03_lot_m1_gives_its_surface_route():
    u = _quantity()
    representation = CellTotal()  # <- total
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Numerics(represent={u: representation})  # <- total-numerics
    message = _refused(caught, surface.NotYetElaborated, "total", traced="total-numerics")
    assert "G03 lot M1" in message and "kind 'total'" in message
    assert "R-b" not in message


def test_a_sample_is_refused_until_the_n16_lot_that_retires_the_table():
    u = _quantity()
    representation = PointValue()  # <- sample
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Numerics(represent={u: representation})
    message = _refused(caught, surface.NotYetElaborated, "sample")
    assert "the N16 lot that retires the transitional table" in message
    assert "kind 'sample'" in message and "R-b" not in message


def test_a_stated_rule_is_refused_until_the_n16_lot_that_retires_the_table():
    u = _quantity()
    representation = CellAverage(quadrature=GaussLegendre(3))  # <- rule
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Numerics(represent={u: representation})
    message = _refused(caught, surface.NotYetElaborated, "rule")
    assert "the N16 lot that retires the transitional table" in message
    assert "with a stated rule" in message and "R-b" not in message


def test_pops_cell_average_with_a_rule_names_the_library_word():
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.CellAverage(quadrature=GaussLegendre(3))  # <- pops-rule
    message = _refused(caught, surface.NotYetElaborated, "pops-rule")
    assert "CellAverage(quadrature=...) of pops.lib.representations" in message
    assert "the N16 lot that retires the transitional table" in message


def test_loads_are_refused_until_n16_lot_2():
    u = _quantity()
    representation = DualLoad()  # <- loads
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Numerics(represent={u: representation})
    assert "N16 lot 2" in _refused(caught, surface.NotYetElaborated, "loads")


@pytest.mark.parametrize("word", [FaceNormalIntegral, EdgeCirculation])
def test_face_and_edge_words_are_refused_until_g08(word):
    u = _quantity()
    representation = word()  # <- face-edge
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Numerics(represent={u: representation})
    assert "G08" in _refused(caught, surface.NotYetElaborated, "face-edge")


def _encode_on_node(node, field):
    return field


def _encode_on_face(face, field):
    return field


def _encode_on_edge(edge, field):
    return field


@pytest.mark.parametrize("carrier, encode, owner", [
    (Node, _encode_on_node, "N16 lot 3"), (Face, _encode_on_face, "G08"),
    (Edge, _encode_on_edge, "G08")])
def test_a_carrier_other_than_the_cell_names_its_lot(carrier, encode, owner):
    @surface.representation
    def carried(cell):
        def decode(coefficients, point):
            return coefficients[0]

        return surface.Coefficients(on=carrier, decode=decode, encode=encode, kind="sample")

    u = _quantity()
    representation = carried()  # <- carrier
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Numerics(represent={u: representation})
    assert owner in _refused(caught, surface.NotYetElaborated, "carrier")


@surface.representation
def modal(cell, rule=None):
    def decode(coefficients, point):
        return coefficients[0] + coefficients[1] * point.reference

    return surface.Coefficients(size=2, decode=decode, pairing=rule)


def test_su9_a_point_dependent_decode_without_a_rule_is_ambiguous():
    u = _quantity()
    representation = modal()  # <- no-rule
    with pytest.raises(surface.AmbiguousElaboration) as caught:
        surface.Numerics(represent={u: representation})
    message = _refused(caught, surface.AmbiguousElaboration, "no-rule",
                       write="pops.Coefficients(..., pairing=Q)")
    assert "(D5d, SU9)" in message


def test_a_point_dependent_decode_with_a_rule_waits_for_n16_lot_2():
    u = _quantity()
    representation = modal(rule=GaussLegendre(2))  # <- modal
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Numerics(represent={u: representation})
    assert "N16 lot 2" in _refused(caught, surface.NotYetElaborated, "modal")
    definition = representation.definition(1)
    assert definition.point_dependent and definition.linear is True


def test_the_projection_under_a_decode_not_proved_linear_is_refused():
    @surface.representation
    def squared(cell):
        def decode(coefficients, point):
            return coefficients[0] * coefficients[0]

        return surface.Coefficients(decode=decode)

    u = _quantity()
    representation = squared()  # <- not-linear
    with pytest.raises(surface.PropertyNotDerived) as caught:
        surface.Numerics(represent={u: representation})
    message = _refused(caught, surface.PropertyNotDerived, "not-linear", write="a decode linear")
    assert "(K4, SV2, IR16)" in message


def test_a_constant_mode_other_than_an_average_or_a_total_waits_for_n16_lot_2():
    @surface.representation
    def doubled(cell):
        def decode(coefficients, point):
            return 2 * coefficients[0]

        return surface.Coefficients(decode=decode)

    u = _quantity()
    representation = doubled()  # <- doubled
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Numerics(represent={u: representation})
    assert "N16 lot 2" in _refused(caught, surface.NotYetElaborated, "doubled")


def test_a_coefficient_outside_the_local_set_is_out_of_scope():
    @surface.representation
    def outside(cell):
        def decode(coefficients, point):
            return coefficients[1]  # <- outside

        return surface.Coefficients(decode=decode)

    u = _quantity()
    representation = outside()
    with pytest.raises(surface.OutOfScope) as caught:
        surface.Numerics(represent={u: representation})  # <- outside-numerics
    # The refusal names the read itself, in decode, and the line where the case is traced.
    _refused(caught, surface.OutOfScope, "outside", write="coefficients[k]",
             traced="outside-numerics")


def test_a_stated_encode_of_several_coefficients_waits_for_n16_lot_2():
    @surface.representation
    def pair(cell):
        def decode(coefficients, point):
            return coefficients[0] + coefficients[1]

        def encode(cell, field):
            return (field, field)

        return surface.Coefficients(size=2, decode=decode, encode=encode)

    u = _quantity()
    representation = pair()  # <- pair
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Numerics(represent={u: representation})
    assert "N16 lot 2" in _refused(caught, surface.NotYetElaborated, "pair")


def test_an_encode_that_does_not_return_size_values_names_size():
    @surface.representation
    def short(cell):
        def decode(coefficients, point):
            return coefficients[0] + coefficients[1]

        def encode(cell, field):
            return [field]

        return surface.Coefficients(size=2, decode=decode, encode=encode)

    u = _quantity()
    with pytest.raises(TypeError, match="size=2 returns 1 values"):
        surface.Numerics(represent={u: short()})


def _decode_of_x(coefficients, x):
    return coefficients[0]


def _decode(coefficients, point):
    return coefficients[0]


def _encode_of_cell(cell, field):
    return field


@pytest.mark.parametrize("decode, encode, named", [
    (_decode_of_x, _encode_of_cell, "def decode(coefficients, point), got def _decode_of_x"),
    (_decode, _encode_on_node, "def encode(cell, field), got def _encode_on_node"),
])
def test_a_decode_or_encode_with_other_parameters_names_the_signature(decode, encode, named):
    @surface.representation
    def misnamed(cell):
        return surface.Coefficients(decode=decode, encode=encode, kind="sample")

    u = _quantity()
    with pytest.raises(TypeError) as caught:
        surface.Numerics(represent={u: misnamed()})
    assert named in str(caught.value)


def test_su5_a_representation_is_always_called():
    u = _quantity()
    with pytest.raises(surface.NotAPopsOperation) as caught:
        surface.Numerics(represent={u: CellAverage})  # <- uncalled
    _refused(caught, surface.NotAPopsOperation, "uncalled", write="CellAverage()")
    with pytest.raises(surface.NotAPopsOperation) as caught:
        surface.Dual(CellAverage)  # <- dual-uncalled
    _refused(caught, surface.NotAPopsOperation, "dual-uncalled", write="pops.Dual(CellAverage())")


def test_su5_options_are_given_by_keyword_and_named():
    with pytest.raises(TypeError, match=r"by keyword, CellAverage\(quadrature=...\)"):
        CellAverage(GaussLegendre(2))
    with pytest.raises(TypeError, match="has the options quadrature, not rule"):
        CellAverage(rule=GaussLegendre(2))
    with pytest.raises(TypeError, match="entity cell first"):
        surface.representation(lambda u: None)
    with pytest.raises(TypeError, match="written bare"):
        surface.representation(name="r")


@pytest.mark.parametrize("options, words", [
    ({"size": 1.0}, "Python integer"),
    ({"size": 0}, "Python integer"),
    ({"on": "cell"}, "carrier of pops.discrete"),
    ({"kind": "load"}, "pops.Dual(R)"),
    ({"pairing": (0.0,)}, "pops.Quadrature"),
    ({"decode": upwind()}, "plain function"),
])
def test_coefficients_check_their_options(options, words):
    def decode(coefficients, point):
        return coefficients[0]

    arguments = {"decode": decode, **options}
    with pytest.raises((TypeError, surface.NotAPopsOperation), match=words.replace("(", r"\(")
                       .replace(")", r"\)").replace(".", r"\.")):
        surface.Coefficients(**arguments)


def test_a_representation_given_where_a_word_is_expected_names_the_construct():
    u = _quantity()

    def decode(coefficients, point):
        return coefficients[0]

    with pytest.raises(TypeError, match="what a @pops.representation returns"):
        surface.Numerics(represent={u: surface.Coefficients(decode=decode)})
    with pytest.raises(TypeError, match="CellAverage\\(\\) of pops.lib.representations"):
        surface.Numerics(represent={u: "average"})


@pytest.mark.parametrize("name", ["CellTotal", "PointValue", "DualLoad", "FaceNormalIntegral",
                                  "EdgeCirculation"])
def test_a_named_representation_is_a_word_of_the_library_not_of_pops(name):
    with pytest.raises(surface.NotYetElaborated) as caught:
        getattr(surface, name)()
    assert "%s() of pops.lib.representations" % name in str(caught.value)
