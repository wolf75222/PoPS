"""SU-T2 (spec 0.3, 2.6.7) for a scheme of one's own, lot 2 of N15 (ROM-98).

Each refusal that ``@pops.scheme``, its entities and the words of ``pops.discrete`` meet in this
lot has a test that triggers it and checks its message: the file and the line of the user's
statement, the rule, and the construct to write (SU11). A scheme runs when ``pops.Numerics``
receives it (SU5), so a refusal raised in its body names the line of the body and the line of the
``Numerics`` call. Each construct of 2.6.4 that a later lot elaborates is ``NotYetElaborated`` and
names that lot. The statement that a refusal names ends with a marker comment, ``# <- name``.
"""
import os

import pytest

import pops.surface as surface
from pops.lib.numerics.schemes import rusanov, two_point, upwind
from pops.lib.time.methods import ssprk2
from pops.surface.discrete import (
    CellValue, FaceFlux, FaceValue, Reconstruct, spectral_radius, sum_faces,
)
from pops.surface.math import ddt, div, dot, grad, largest

HERE = os.path.abspath(__file__)


def _line(marker):
    lines = [number for number, text in enumerate(open(HERE, encoding="utf-8"), start=1)
             if text.rstrip().endswith("# <- " + marker)]
    assert len(lines) == 1, "marker %r is not on exactly one line" % marker
    return lines[0]


def _at(error, marker):
    return (error.site.file, error.site.line) == (HERE, _line(marker))


def _model(value=lambda u: u**2 / 2, diffusive=False):
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("burgers")
    u = model.quantity("u", surface.Scalar(domain))
    flux = model.flux(of=u, value=value(u))
    terms = flux
    diffusion = None
    if diffusive:
        nu = model.parameter("nu", default=1e-2, admissible=surface.Positive())
        diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
        terms = flux + diffusion
    model.balance(ddt(u) == -div(terms))
    return domain, model, u, flux, diffusion


def _numerics(u, flux, scheme):
    return surface.Numerics(represent={u: surface.CellAverage()}, schemes={flux: scheme})


@surface.scheme
def central(face, term):
    q = term.quantity
    return FaceFlux((dot(term.law(states={q: face.left(q)}), face.normal)
                     + dot(term.law(states={q: face.right(q)}), face.normal)) / 2)


# --- @pops.scheme and its call (SU5) --------------------------------------------------------------

def test_an_uncalled_scheme_is_refused():
    _, _, u, flux, _ = _model()
    with pytest.raises(surface.NotAPopsOperation) as caught:
        surface.Numerics(represent={u: surface.CellAverage()}, schemes={flux: central})  # <- uncalled
    assert _at(caught.value, "uncalled")
    assert "Write central()" in str(caught.value)


def test_an_unknown_option_is_a_python_error_at_the_call():
    with pytest.raises(TypeError, match="central has the options none, not limiter"):
        central(limiter=1)


def test_an_option_is_given_by_keyword():
    @surface.scheme
    def scaled(face, term, factor):
        q = term.quantity
        return FaceFlux(factor * face.left(q))

    with pytest.raises(TypeError, match=r"needs the options factor, given by keyword"):
        scaled()
    with pytest.raises(TypeError, match=r"takes its options by keyword, scaled\(factor=...\)"):
        scaled(2)


def test_the_decorator_is_written_bare():
    with pytest.raises(TypeError, match="written bare"):
        surface.scheme(name="x")


def test_a_scheme_receives_term():
    with pytest.raises(TypeError, match="takes term"):
        @surface.scheme
        def no_term(face):
            return FaceFlux(0)


def test_a_scheme_of_a_later_entity_names_its_lot():
    with pytest.raises(surface.NotYetElaborated, match="edge fluxes of G06"):
        @surface.scheme
        def on_edges(edge, term):
            return None


def test_a_cell_scheme_without_axis_names_its_lot():
    with pytest.raises(surface.NotYetElaborated, match="CellValue, CellRate or CellResidual"):
        @surface.scheme
        def pointwise(cell, term):
            return None


def test_a_scheme_that_returns_the_wrong_builder_is_refused():
    @surface.scheme
    def wrong(face, term):
        return Reconstruct(face.left(term.quantity))

    _, _, u, flux, _ = _model()
    with pytest.raises(surface.RealizationMismatch, match="Write FaceFlux"):
        _numerics(u, flux, wrong())


def test_a_cell_scheme_covers_no_key():
    @surface.scheme
    def centre(cell, term, axis):
        return Reconstruct(cell.value(term.quantity))

    _, _, u, flux, _ = _model()
    with pytest.raises(surface.RealizationMismatch, match="reconstruction= of a face scheme"):
        _numerics(u, flux, centre())
    with pytest.raises(TypeError, match="takes a scheme that returns Reconstruct"):
        FaceFlux(0, reconstruction=central())


# --- term.expect (CC2) ----------------------------------------------------------------------------

@surface.scheme
def linear_upwind(face, term):
    q = term.quantity
    unit_flux = term.law(states={q: 1})
    term.expect(unit_flux * q)  # <- expect
    s = dot(unit_flux, face.normal)
    return FaceFlux(s * face.left(q), speed=s)


def test_term_expect_refuses_a_key_of_another_form():
    _, _, u, flux, _ = _model()
    with pytest.raises(surface.RealizationMismatch) as caught:
        represent = {u: surface.CellAverage()}
        surface.Numerics(represent=represent, schemes={flux: linear_upwind()})  # <- expect-numerics
    message = str(caught.value)
    assert _at(caught.value, "expect")
    assert "the scheme expects the key u**2 / 2 to be 1**2 / 2 * u" in message
    assert "The case is traced at %s:%d" % (HERE, _line("expect-numerics")) in message


def test_term_expect_accepts_the_key_of_its_form():
    _, _, u, flux, _ = _model(lambda u: 3 * u)
    _numerics(u, flux, linear_upwind())


# --- the scalars and the vectors of a face (2.6.3, review Q2) ---------------------------------------

def test_dot_of_a_scalar_with_the_normal_is_a_support_mismatch():
    @surface.scheme
    def scalar_dot(face, term):
        q = term.quantity
        return FaceFlux(dot(face.left(q), face.normal))  # <- scalar-dot

    _, _, u, flux, _ = _model()
    with pytest.raises(surface.SupportMismatch) as caught:
        _numerics(u, flux, scalar_dot())
    assert _at(caught.value, "scalar-dot")
    assert "Write dot(term.law(states=...), face.normal)" in str(caught.value)


def test_the_flux_of_the_key_where_a_scalar_is_taken_is_a_support_mismatch():
    @surface.scheme
    def vector_value(face, term):
        q = term.quantity
        return FaceFlux(term.law(states={q: face.left(q)}))

    _, _, u, flux, _ = _model()
    with pytest.raises(surface.SupportMismatch, match="its normal component"):
        _numerics(u, flux, vector_value())


def test_another_dot_belongs_to_n11():
    @surface.scheme
    def other_dot(face, term):
        q = term.quantity
        return FaceFlux(dot(face.left(q), face.right(q)))

    _, _, u, flux, _ = _model()
    with pytest.raises(surface.NotYetElaborated, match="N11"):
        _numerics(u, flux, other_dot())


def test_largest_takes_a_list():
    with pytest.raises(TypeError, match="a tuple is a point"):
        largest((1, 2))


def test_largest_in_a_value_of_the_model_belongs_to_n11():
    domain, model, u, flux, _ = _model(lambda u: largest([u, 0 * u]))
    numerics = _numerics(u, flux, rusanov())
    u0 = model.data("u0", surface.Scalar(domain))
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    with pytest.raises(surface.NotYetElaborated, match="N11"):
        surface.Case(model=model, numerics=numerics, program=program,
                     layout={domain: surface.Uniform(cells=8)})


def test_a_speed_that_is_not_derived_is_ambiguous():
    @surface.scheme
    def radius(face, term):
        q = term.quantity
        s = spectral_radius(term, face.left(q), face.normal)  # <- radius
        return FaceFlux(s * face.left(q), speed=s)

    _, _, u, flux, _ = _model(lambda u: u * abs(u))
    with pytest.raises(surface.AmbiguousElaboration) as caught:
        _numerics(u, flux, radius())
    assert _at(caught.value, "radius")
    assert "Write FaceFlux(..., speed=s)" in str(caught.value)


# --- the constructs of later lots (lot note, §3) ----------------------------------------------------

def _face_reading(read):
    @surface.scheme
    def reading(face, term):
        return FaceFlux(read(face, term))

    return reading


@pytest.mark.parametrize("read, owner", [
    (lambda face, term: face.distance, r"spacing \(C31\)"),
    (lambda face, term: face.measure, "geometry reads"),
    (lambda face, term: face.normal_gradient(term.quantity), r"spacing \(C31\)"),
    (lambda face, term: face.left_cell.shift(1), "stencil faces of N7a"),
    (lambda face, term: face.left_cell.width, "geometry reads"),
    (lambda face, term: term.law(gradients={}), r"spacing \(C31\)"),
    (lambda face, term: FaceValue(0), "keys that this builder covers"),
    (lambda face, term: CellValue(0), "keys that this builder covers"),
    (lambda face, term: sum_faces(face, abs), "reductions"),
], ids=["distance", "measure", "normal_gradient", "left_cell.shift", "left_cell.width",
        "law-gradients", "FaceValue", "CellValue", "sum_faces"])
def test_a_construct_of_a_later_lot_names_its_owner(read, owner):
    _, _, u, flux, _ = _model()
    with pytest.raises(surface.NotYetElaborated, match=owner):
        _numerics(u, flux, _face_reading(read)())


def test_per_side_contributions_name_their_lot():
    with pytest.raises(surface.NotYetElaborated, match="per-side contributions"):
        FaceFlux(0, left=1)


def test_a_face_scheme_under_a_diffusive_flux_names_its_lot():
    _, _, u, flux, diffusion = _model(diffusive=True)
    with pytest.raises(surface.NotYetElaborated, match="spacing"):
        surface.Numerics(represent={u: surface.CellAverage()},
                         schemes={flux: rusanov(), diffusion: central()})


def test_a_cell_read_of_geometry_names_its_lot():
    @surface.scheme
    def wide(cell, term, axis):
        return Reconstruct(cell.value(term.quantity) * cell.width)

    @surface.scheme
    def face_with(face, term, reconstruction=None):
        q = term.quantity
        return FaceFlux(face.left(q), reconstruction=reconstruction)

    _, _, u, flux, _ = _model()
    with pytest.raises(surface.NotYetElaborated, match="geometry reads"):
        _numerics(u, flux, face_with(reconstruction=wide()))


def test_a_speed_per_factor_names_the_product_supports():
    with pytest.raises(surface.NotYetElaborated, match="product supports of G09"):
        upwind(speed={})


def test_a_stated_speed_is_a_plain_function_of_face():
    with pytest.raises(surface.NotAPopsOperation, match="plain function of face"):
        upwind(speed=central())
    with pytest.raises(TypeError, match="takes a function of face"):
        upwind(speed=1.0)


def test_a_library_reconstruction_is_a_cell_scheme():
    with pytest.raises(TypeError, match="takes a scheme that returns Reconstruct"):
        rusanov(reconstruction=two_point())


# --- SU7, SU13: the entities exist only as parameters --------------------------------------------

def test_an_entity_outside_its_function_is_a_name_error():
    with pytest.raises(NameError):
        face.normal  # noqa: F821, B018


def test_pops_discrete_exports_no_entity():
    import pops.surface.discrete as discrete

    with pytest.raises(surface.SymbolNotInSurface, match="parameters of a scheme"):
        discrete.face  # noqa: B018
    with pytest.raises(ImportError):
        from pops.surface.discrete import cell  # noqa: F401


# --- A1: largest and smallest in a face that a method derives -------------------------------------

def test_no_method_of_this_lot_derives_a_face():
    """A face that reads largest or smallest lowers them to b6 maximum and minimum (SU-T3), and no
    method of level 1 in this lot derives a face: an implicit step needs pops.solve, which a later
    lot elaborates. When it does, such a face keeps b6's behaviour or names the lot 2 of N11."""
    with pytest.raises(surface.NotYetElaborated, match="solves of N9"):
        surface.solve()
    with pytest.raises(surface.NotYetElaborated, match="solves of N9"):
        surface.unknown("x")
