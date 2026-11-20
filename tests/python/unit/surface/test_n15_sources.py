"""N15 FaceFlux/CellRate lot FC-1 (ROM-98, note §3): ``model.source``, ``CellRate`` and
``pointwise()``.

A source of one quantity, ``model.source(of=u, value=e)``, enters its balance as ``+ S``, ``- S``
or scaled by an exact number (§3.7). It is covered by a scheme over ``cell`` that returns
``CellRate``, such as ``pointwise()`` of ``pops.lib.numerics``: its value is read at the centre of
the cell under one constant mode, the rate itself under ``CellAverage()``, μ(K) times it under
``CellTotal()`` (N16 5.1, A1). The case elaborates on the split route: a b6 named source, its
selection realized by ``StateStorage()``.

- SU-T3: the surface case gives the program IR, the artifact, the semantic identity and the
  generated C++ of the same case written by hand for b6 and N8.
- Under ``CellTotal()`` the source is decoded and encoded by the row of G03 lot R-b, like a rate.
- SU-T2: each refusal of the lot names the user's line (``# <- name``), the rule or the lot.
"""
import json
import os

import pytest

import pops
import pops.surface as surface
from pops import math as pm
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.numerics import pointwise, upwind
from pops.lib.representations import CellAverage, CellTotal, PointValue
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid, PeriodicAxes
from examples.api_v02 import b6_numerics
from pops.numerics import DiscretizationPlan, StateStorage
from pops.params import ConstParam, RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.surface.discrete import CellRate, CellValue, FaceFlux
from pops.surface.math import ddt, div, dot, exp, grad, vector
from pops.time._program.elaboration import Advance, Proposal, Retry, elaborate
from pops.time._program.traced_method import Candidate, method

HERE = os.path.abspath(__file__)
CELLS = 16
LENGTH = 3.0


def _line(marker):
    lines = [number for number, text in enumerate(open(HERE, encoding="utf-8"), start=1)
             if text.rstrip().endswith("# <- " + marker)]
    assert len(lines) == 1, "marker %r is not on exactly one line" % marker
    return lines[0]


def _at(error, marker):
    return (error.site.file, error.site.line) == (HERE, _line(marker))


DOMAIN = surface.Box(x=surface.Periodic(0, LENGTH))


def _model(name="decaying_advection"):
    domain = DOMAIN
    model = surface.Model(name)
    u = model.quantity("u", surface.Scalar(domain))
    return domain, model, u


def _decay(representation=None, sign=1, scheme=pointwise, name=None):
    """du/dt = -div(a u) + sign S, S = -k u, under ``pointwise()``, ssprk2 and a fixed step."""
    domain, model, u = _model()
    a = model.constant("a", value=1.0)
    k = model.parameter("k", default=2.0)
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=a * u)
    decay = model.source(of=u, value=-k * u, name=name)
    model.balance(ddt(u) == -div(advection) + (decay if sign == 1 else -decay))
    numerics = surface.Numerics(represent={u: (representation or CellAverage)()},
                                schemes={advection: upwind(), decay: scheme()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3, attempts=2),
                              initial={u: u0})
    return surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)})


def _n8_ssprk2():
    """The ssprk2 of ``pops.lib.time.methods`` written for N8: ``B`` the step, ``p.dt``."""

    @method
    def ssprk2(B, p, rate):
        u, t, h = B.state, B.time, p.dt
        first = u + h * rate(u, t)
        return Candidate((u + first + h * rate(first, t + h)) / 2, t + h)

    return ssprk2


def _hand(sign=1):
    """The same case written for b6 and N8: the b6 ``rusanov`` of the transport, a named source
    whose value is the law at the cell, and its selection realized by ``StateStorage()``."""
    frame = CartesianDomain("box", lower=(0.0,), upper=(LENGTH,)).frame(Cartesian1D())
    (x,) = frame.axes
    model = pops.Model("decaying_advection", frame=frame)
    u = model.state("u", components=("u",))
    a = model.value(model.param(ConstParam("a", 1.0)))
    k = model.value(model.param(RuntimeParam("k", default=2.0)))
    flux_u = model.flux("flux_u", state=u, frame=frame, components={x: (a * u[0],)},
                        waves={x: (a,)})
    source_u = model.source("source_u", on=u, value=((-k) * u[0],))
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u)
                      + (source_u if sign == 1 else -source_u))
    plan = DiscretizationPlan()
    advection = rate.select(flux_u)
    plan.rates.add(advection, b6_numerics.rusanov(flux_u, u))
    source = rate.select(source_u)
    plan.rates.add(source, StateStorage())
    case = pops.Case("decaying_advection")
    block = case.block("decaying_advection", model)
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))

    def transport(state):
        # A selection carries the sign of its term in the rate of b6.
        return advection(state) + source(state)

    elaborate(Advance(_n8_ssprk2()(rate=transport), retry=Retry(max_attempts=2)), case=case,
              state=block[u], proposal=Proposal(0.001))
    layout = Uniform(CartesianGrid(frame=frame, cells=(CELLS,), periodic=PeriodicAxes((x,))))
    return case, layout


def _without_provenance(value):
    if isinstance(value, dict):
        return {key: _without_provenance(item) for key, item in value.items()
                if key != "provenance"}
    if isinstance(value, list):
        return [_without_provenance(item) for item in value]
    return value


def _forms(case, layout):
    resolved = pops.resolve(pops.validate(case), layout=layout)
    return {
        "program": case._time._serialize(include_provenance=False),
        "artifact": _without_provenance(json.loads(resolved.snapshot._artifact_canonical_json)),
        "semantic_identity": resolved.snapshot.semantic_identity,
        "cpp": emit_cpp_program(resolved.time, model_graph=ProgramModelGraph
                                .from_resolved_blocks(resolved.blocks)),
    }


def _surface_forms(case):
    elaborated = case.elaborated
    return _forms(elaborated.case, elaborated.layout)


# --- SU-T3 ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("sign", [1, -1])
def test_su_t3_a_source_under_pointwise_gives_the_forms_of_the_hand_written_case(sign):
    """``+ S`` and ``- S``: the split route of b6, a named source selected and realized by
    ``StateStorage()``, with the same program, artifact, identity and C++ (§3.4)."""
    surface_forms, hand_forms = _surface_forms(_decay(sign=sign)), _forms(*_hand(sign=sign))
    for form in ("program", "artifact", "semantic_identity", "cpp"):
        assert surface_forms[form] == hand_forms[form], form


def test_a_sample_gives_the_program_and_cpp_of_the_average():
    """Under ``PointValue()`` the value read at the centre is the rate itself, as under an
    average (N16 5.1): the program and the C++ are those of ``CellAverage()``."""
    average, sample = _surface_forms(_decay()), _surface_forms(_decay(PointValue))
    assert sample["program"] == average["program"]
    assert sample["cpp"] == average["cpp"]
    assert sample["semantic_identity"] != average["semantic_identity"]


def test_a_total_decodes_before_the_source_and_encodes_its_rate():
    """Under ``CellTotal()`` the named source reads ū = U/μ(K) and its rate is multiplied by μ(K),
    by the row of G03 lot R-b, as the rate of the transport (note §3.4, A1)."""
    cpp = _surface_forms(_decay(CellTotal))["cpp"]
    lines = [line.strip() for line in cpp.splitlines()]
    scratch = next(number for number, line in enumerate(lines)
                   if "rhs_scratch(" in line and "source" in "".join(lines[number - 3:number]))
    (decode,) = [number for number in range(scratch - 3, scratch)
                 if lines[number].startswith("ctx.decode_cell_total(")]
    density = lines[decode].split(",")[1].split(")")[0].strip()
    assert density in lines[scratch]
    output = lines[scratch].split("&")[1].split("=")[0].strip()
    assert "ctx.encode_cell_total_rate(%s);  // G03 R-b: the rate of the total" % output in lines
    assert cpp.count("ctx.decode_cell_total(") == 4  # the transport and the source, two stages


# --- inspect and names ----------------------------------------------------------------------------

def _rows(case, statement):
    return [row for row in surface.inspect(case).rows if row.construct.startswith(statement)]


def test_inspect_shows_the_source_its_scheme_and_the_kind_of_its_value():
    (row,) = _rows(_decay(), "model.source(")
    assert row.elaborated == "source_u = model.source('source_u', on=u, value=((-(k) * u[0]),))"
    choices = {choice.name: (choice.value, choice.origin) for choice in row.choices}
    assert choices["name"] == ("source_u", "derived")
    assert choices["scheme"][1] == "stated"
    assert choices["kind of value"] == (
        "a value read at the centre of the cell, neither a load nor a total", "derived")
    assert choices["delivered"] == ("the rate of CellAverage(): the value read at the cell",
                                    "derived")
    (total,) = _rows(_decay(CellTotal), "model.source(")
    delivered = dict((choice.name, choice.value) for choice in total.choices)["delivered"]
    assert "μ(K) times the value read at the cell" in delivered


def test_inspect_shows_the_selection_realized_by_state_storage():
    (row,) = [row for row in surface.inspect(_decay()).rows
              if "u.select(source_u)" in row.elaborated]
    assert row.elaborated.startswith("plan.rates.add(u.select(source_u), StateStorage()")


def test_a_named_source_keeps_its_name():
    (row,) = _rows(_decay(name="decay"), "model.source(")
    assert row.elaborated.startswith("decay = model.source('decay', on=u")


def test_the_word_pointwise_is_the_gallery_scheme():
    """``pointwise()`` of ``pops.lib.numerics`` is the scheme of the gallery of 2.6.4: the law of
    the source at the cell value, ``CellRate(term.law(states={q: cell.value(q)}))``."""
    from pops.lib.numerics import cell_rates

    assert pointwise is cell_rates.pointwise
    assert "pointwise" in pops.lib.numerics.__all__


# --- a scheme of one's own over cell --------------------------------------------------------------

@surface.scheme
def doubled(cell, term):
    q = term.quantity
    return CellRate(2 * term.law(states={q: cell.value(q)}))


def test_a_scheme_of_ones_own_over_cell_gives_its_cell_rate():
    (row,) = _rows(_decay(scheme=doubled), "model.source(")
    assert "value=((2 * (-(k) * u[0])),))" in row.elaborated


@surface.scheme
def reads_x(cell, term):
    return CellRate(DOMAIN.x)


@surface.scheme
def loads(cell, term):
    q = term.quantity
    return CellRate(cell.load(term.law(states={q: cell.value(q)})))


@surface.scheme
def integrates(cell, term):
    q = term.quantity
    return CellRate(cell.integral(term.law(states={q: cell.value(q)})))


@surface.scheme
def shifted(cell, term):
    q = term.quantity
    return CellRate(cell.shift(1).value(q))


@surface.scheme
def a_face_scheme(face, term):
    q = term.quantity
    return FaceFlux(dot(term.law(states={q: face.left(q)}), face.normal))


@surface.scheme
def a_cell_value(cell, term):
    q = term.quantity
    return CellValue(cell.value(q))


def _decay_with(scheme):
    domain, model, u = _model()
    k = model.parameter("k", default=2.0)
    decay = model.source(of=u, value=-k * u)
    model.balance(ddt(u) == decay)
    return u, decay


def test_a_coordinate_and_a_load_are_read_by_a_cell_rate():
    """``pops.Numerics`` runs both schemes: FC-2 types and elaborates a coordinate read at the
    centre of the cell and a load (``test_n15_cell_load.py``)."""
    u, decay = _decay_with(reads_x)
    surface.Numerics(represent={u: CellAverage()}, schemes={decay: reads_x()})
    u, decay = _decay_with(loads)
    surface.Numerics(represent={u: CellAverage()}, schemes={decay: loads()})


def test_an_integral_waits_for_n16_lot_2():
    u, decay = _decay_with(integrates)
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Numerics(represent={u: CellAverage()}, schemes={decay: integrates()})  # <- integral
    assert "N16 lot 2" in str(caught.value)


def test_a_neighbour_read_by_a_cell_rate_waits_for_the_region_form():
    u, decay = _decay_with(shifted)
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Numerics(represent={u: CellAverage()}, schemes={decay: shifted()})  # <- shift
    assert "N16" in str(caught.value)


def test_a_source_does_not_take_a_face_scheme():
    u, decay = _decay_with(a_face_scheme)
    with pytest.raises(surface.RealizationMismatch) as caught:
        surface.Numerics(represent={u: CellAverage()}, schemes={decay: a_face_scheme()})
    assert "is a Source" in str(caught.value)


def test_a_cell_value_waits_for_its_lot():
    """A scheme over cell that returns ``CellValue`` is a later builder of 2.6.4."""
    u, decay = _decay_with(a_cell_value)
    with pytest.raises(surface.NotYetElaborated, match="CellValue"):
        surface.Numerics(represent={u: CellAverage()}, schemes={decay: a_cell_value()})


def test_a_flux_does_not_take_pointwise():
    domain, model, u = _model()
    flux = model.flux(of=u, value=u)
    model.balance(ddt(u) == -div(flux))
    with pytest.raises((TypeError, ValueError, surface.SurfaceRefusal)) as caught:
        surface.Numerics(represent={u: CellAverage()}, schemes={flux: pointwise()})
    assert "pointwise" in str(caught.value)


# --- SU-T2: the refusals of model.source and of the balance ---------------------------------------

def test_a_source_over_a_support_waits_for_r4b():
    _, model, u = _model()
    with pytest.raises(surface.NotYetElaborated) as caught:
        model.source(of=u, value=-u, support=1)  # <- support
    assert _at(caught.value, "support")
    assert "R4b" in str(caught.value)


def test_a_source_that_reads_another_quantity_waits_for_the_record_states():
    domain, model, u = _model()
    v = model.quantity("v", surface.Scalar(domain))
    with pytest.raises(surface.NotYetElaborated) as caught:
        model.source(of=u, value=-u * v)  # <- other
    assert _at(caught.value, "other")
    assert "record states of N8" in str(caught.value)


def test_a_source_that_reads_a_gradient_waits_for_the_scheme_gradients():
    _, model, u = _model()
    with pytest.raises(surface.NotYetElaborated) as caught:
        model.source(of=u, value=grad(u))  # <- grad
    assert _at(caught.value, "grad")
    assert "GR" in str(caught.value)


def test_a_vector_source_of_a_scalar_quantity_is_a_support_mismatch():
    _, model, u = _model()
    with pytest.raises(surface.SupportMismatch) as caught:
        model.source(of=u, value=vector(u, u))  # <- vector
    assert _at(caught.value, "vector")


def test_a_source_whose_value_reads_a_term_is_refused():
    _, model, u = _model()
    flux = model.flux(of=u, value=u)
    with pytest.raises(TypeError, match="a term enters a balance"):
        model.source(of=u, value=flux)


def test_a_flux_outside_div_names_its_divergence():
    _, model, u = _model()
    flux = model.flux(of=u, value=u)
    with pytest.raises(TypeError, match=r"-div\(") as caught:
        model.balance(ddt(u) == flux)
    assert "outside div" in str(caught.value)


def test_a_term_under_a_function_is_not_additive():
    _, model, u = _model()
    decay = model.source(of=u, value=-u)
    with pytest.raises(surface.TermNotAdditive) as caught:
        model.balance(ddt(u) == exp(decay))  # <- exp
    assert _at(caught.value, "exp")
    assert "exp" in str(caught.value)


def test_a_product_of_two_terms_is_not_additive():
    _, model, u = _model()
    decay = model.source(of=u, value=-u)
    growth = model.source(of=u, value=u)
    with pytest.raises(surface.TermNotAdditive) as caught:
        model.balance(ddt(u) == decay * growth)  # <- product
    assert _at(caught.value, "product")


def test_inline_algebra_waits_for_the_anonymous_source():
    _, model, u = _model()
    k = model.parameter("k", default=2.0)
    flux = model.flux(of=u, value=u)
    with pytest.raises(surface.NotYetElaborated) as caught:
        model.balance(ddt(u) == -div(flux) - k * u)  # <- inline
    assert _at(caught.value, "inline")
    assert "anonymous sources of §3.7" in str(caught.value)
    assert "model.source(of=u" in str(caught.value)


@pytest.mark.parametrize("weight", ["twice", "half"])
def test_a_source_weighted_other_than_one_is_refused_by_the_weight_rule(weight):
    """``2 * S`` and ``S / 2`` are occurrences of weight 2 and 1/2 (§3.7), refused by the weight
    rule of this lot, not as a term that is not additive."""
    domain, model, u = _model()
    u0 = model.data("u0", surface.Scalar(domain))
    decay = model.source(of=u, value=-u)
    model.balance(ddt(u) == (2 * decay if weight == "twice" else decay / 2))  # <- weighted
    numerics = surface.Numerics(represent={u: CellAverage()}, schemes={decay: pointwise()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Case(model=model, numerics=numerics, program=program,
                     layout={domain: surface.Uniform(cells=CELLS)})
    assert _at(caught.value, "weighted")
    assert ("the weight %s of -u in the balance of u"
            % ("2" if weight == "twice" else "1/2")) in str(caught.value)


def test_a_source_without_a_scheme_is_not_covered():
    """A source takes no default scheme (D5): a case that leaves it out of ``schemes=`` is
    refused, naming it."""
    domain, model, u = _model()
    u0 = model.data("u0", surface.Scalar(domain))
    decay = model.source(of=u, value=-u)
    model.balance(ddt(u) == decay)
    numerics = surface.Numerics(represent={u: CellAverage()}, schemes={})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    with pytest.raises(surface.CoverageNotExact) as caught:
        surface.Case(model=model, numerics=numerics, program=program,
                     layout={domain: surface.Uniform(cells=CELLS)})
    assert "-u" in str(caught.value)
