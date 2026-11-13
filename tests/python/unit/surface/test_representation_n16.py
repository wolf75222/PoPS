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

- The table lifted for three words (N16 5.1, A1): ``CellTotal()``, ``PointValue()`` and
  ``CellAverage(quadrature=GaussLegendre(3))``, of the library or written in the case, take the row
  of b6 that serves them (``plan.representations.add``); a sample and a stated rule emit the program
  and the C++ of the average, a total emits R-b's decode and encode, and an initial value other than
  a datum waits for N16 lot 2 (the projection by encode). The oracle tests on a graded layout run
  natively in ``tests/python/integration/runtime/test_n16_three_words_native.py``, and the surface
  case in ``test_surface_n16_three_words_native.py``.
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

def _l1(representation, diffusion=True):
    """L1 of §17.1; ``diffusion=False`` keeps its advection only, the case that a total runs (R-b
    refuses the StencilFlux of two_point() under a total until a case runs it)."""
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("advection_diffusion")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    nu = model.parameter("nu", default=1e-2, admissible=surface.Positive())
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=a * u)
    schemes = {advection: upwind()}
    if diffusion:
        diffusive = model.diffusive_flux(of=u, value=-nu * grad(u))
        model.balance(ddt(u) == -div(advection + diffusive))
        schemes[diffusive] = two_point()
    else:
        model.balance(ddt(u) == -div(advection))
    numerics = surface.Numerics(represent={u: representation}, schemes=schemes)
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


def test_pops_cell_average_with_a_rule_names_the_library_word():
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.CellAverage(quadrature=GaussLegendre(3))  # <- pops-rule
    message = _refused(caught, surface.NotYetElaborated, "pops-rule")
    assert "CellAverage(quadrature=...) of pops.lib.representations" in message
    assert "R-b" not in message and "retires the transitional table" not in message


# --- the table lifted for three words (N16 5.1, A1; the lot that lifts the table) ---------------

_THREE = {
    "total": (CellTotal, case_total, contract.cell_total),
    "sample": (PointValue, case_sample, contract.point_value),
    "rule": (lambda: CellAverage(quadrature=GaussLegendre(3)),
             lambda: case_average_with_rule(rule=GaussLegendre(3)),
             lambda rank: contract.cell_average(rank, quadrature=GaussLegendre(3))),
}


@pytest.fixture(scope="module")
def three_forms():
    return {(word, spelling): _forms(_l1(_THREE[word][0 if spelling == "library" else 1](),
                                         diffusion=word != "total"))
            for word in _THREE for spelling in ("library", "case")}


@pytest.fixture(scope="module")
def advection_forms():
    return _forms(_l1(surface.CellAverage(), diffusion=False))


@pytest.mark.parametrize("spelling", ["library", "case"])
@pytest.mark.parametrize("word", sorted(_THREE))
def test_a_word_of_one_constant_mode_takes_its_row_of_b6(word, spelling):
    """CellTotal(), PointValue() and CellAverage(quadrature=GaussLegendre(3)), of the library or
    written in the case, elaborate to the row plan.representations.add(state, definition) of b6,
    matched by the whole identity (N16 5.5); the published arrays carry that identity (K4)."""
    library, written, oracle = _THREE[word]
    case = _l1(library() if spelling == "library" else written(), diffusion=word != "total")
    elaborated = case.elaborated
    (quantity,) = tuple(elaborated.states)
    resolved = pops.resolve(pops.validate(elaborated.case), layout=elaborated.layout)
    numerics = resolved.blocks[0].numerics
    (row,) = numerics.representations
    assert row.subject.local_id == quantity.name
    assert numerics.representation_of(row.subject) == row.method == oracle(1)
    assert elaborated.representations[quantity].definition == oracle(1)
    assert numerics.cell_total_states() == (() if word != "total" else
                                            (numerics.representations[0].subject,))


@pytest.mark.parametrize("word", ["sample", "rule"])
def test_a_sample_and_a_stated_rule_emit_the_program_and_cpp_of_the_average(word, three_forms,
                                                                           b6_forms):
    """MG1: a sample and a stated rule have the rate of an average (N16 5.1), so the program IR
    and the generated C++ are those of pops.CellAverage(); the artifact and the semantic identity
    carry the row, so they differ."""
    for spelling in ("library", "case"):
        forms = three_forms[(word, spelling)]
        assert forms["program"] == b6_forms["program"]
        assert forms["cpp"] == b6_forms["cpp"]
        assert forms["semantic_identity"] != b6_forms["semantic_identity"]
    assert three_forms[(word, "library")] == three_forms[(word, "case")]


def test_a_total_emits_its_decode_and_encode(three_forms, advection_forms):
    """A total is lowered by R-b's decode ū = U/μ(K) and encode μ(K) R; the word and the same
    written in the case give the same forms."""
    forms = three_forms[("total", "library")]
    assert forms == three_forms[("total", "case")]
    assert forms["program"] == advection_forms["program"]
    assert forms["cpp"] != advection_forms["cpp"]
    assert "decode_cell_total" in forms["cpp"] and "encode_cell_total_rate" in forms["cpp"]
    assert "decode_cell_total" not in advection_forms["cpp"]


def test_a_total_under_the_stencil_flux_of_two_point_is_refused_by_r_b():
    """The StencilFlux of two_point() is not among the rates that read the decoded density under
    a total (G03 lot R-b), so L1 with its diffusion resolves to that refusal, before any C++."""
    from pops.codegen._cell_total_consumers import CellTotalNotYetElaborated

    elaborated = _l1(CellTotal()).elaborated
    with pytest.raises(CellTotalNotYetElaborated) as caught:
        pops.resolve(pops.validate(elaborated.case), layout=elaborated.layout)
    assert caught.value.reason == "cell_total_operation"
    assert "stencil_flux_divergence" in str(caught.value)


def test_inspect_reports_the_row_of_the_word():
    rows = [row for row in surface.inspect(_l1(CellTotal(), diffusion=False)).rows
            if row.construct.startswith("represent=")]
    assert len(rows) == 1 and "plan.representations.add(u, the definition of kind 'total'" \
        in rows[0].elaborated


@pytest.mark.parametrize("word", sorted(_THREE))
def test_an_initial_value_projected_under_a_word_waits_for_n16_lot_2(word):
    """The analytic route of b6 projects cell averages; the projection by the encode of a total,
    a sample or a stated rule is N16 lot 2's, so a value other than a datum is refused at the
    Program, and a datum (the coefficients, at bind) is the route."""
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("m")
    u = model.quantity("u", surface.Scalar(domain))
    advection = model.flux(of=u, value=1.0 * u)
    model.balance(ddt(u) == -div(advection))
    numerics = surface.Numerics(represent={u: _THREE[word][0]()}, schemes={advection: upwind()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),  # <- initial
                              initial={u: 1.0})
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Case(model=model, numerics=numerics, program=program,  # <- initial-case
                     layout={domain: surface.Uniform(cells=8)})
    message = _refused(caught, surface.NotYetElaborated, "initial", traced="initial-case")
    assert "N16 lot 2 (the projection by encode" in message
    assert "pops.Program(initial={u: d}), d = model.data(...)" in message
    assert "the coefficients bound at pops.bind" in message


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
