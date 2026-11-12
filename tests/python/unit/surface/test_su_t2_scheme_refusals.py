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
from pops.lib.numerics import rusanov, two_point, upwind
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
    _, _, u, flux, _ = _model(lambda u: largest([u, 0 * u]))
    with pytest.raises(surface.NotYetElaborated, match="N11"):
        _numerics(u, flux, rusanov())


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
    (lambda face, term: face.distance, r"transport flux and reads the spacing \(C31\)"),
    (lambda face, term: face.measure, "geometry reads"),
    (lambda face, term: face.normal_gradient(term.quantity),
     r"transport flux and reads the spacing \(C31\)"),
    (lambda face, term: face.left_cell.shift(1), "stencil faces of N7a"),
    (lambda face, term: face.left_cell.width, "geometry reads"),
    (lambda face, term: FaceValue(0), "keys that this builder covers"),
    (lambda face, term: CellValue(0), "keys that this builder covers"),
    (lambda face, term: sum_faces(face, abs), "reductions"),
], ids=["distance", "measure", "normal_gradient", "left_cell.shift", "left_cell.width",
        "FaceValue", "CellValue", "sum_faces"])
def test_a_construct_of_a_later_lot_names_its_owner(read, owner):
    _, _, u, flux, _ = _model()
    with pytest.raises(surface.NotYetElaborated, match=owner):
        _numerics(u, flux, _face_reading(read)())


def test_per_side_contributions_name_their_lot():
    with pytest.raises(surface.NotYetElaborated, match="per-side contributions"):
        FaceFlux(0, left=1)


def test_a_transport_face_under_a_diffusive_flux_is_of_another_kind():
    """N3 lot 1: a diffusive key is realized (two_point() and a scheme of one's own that reads
    term.law(gradients=...)); a face that reads the law of a transport flux under it is refused as
    a scheme of another kind, at the line of its body."""
    _, _, u, flux, diffusion = _model(diffusive=True)
    with pytest.raises(surface.RealizationMismatch) as caught:
        surface.Numerics(represent={u: surface.CellAverage()},
                         schemes={flux: rusanov(), diffusion: central()})
    assert "the law of a transport flux" in str(caught.value)
    assert "Write a scheme of the kind of the key" in str(caught.value)


def test_a_diffusive_face_under_a_transport_flux_is_of_another_kind():
    @surface.scheme
    def gradient_law(face, term):
        q = term.quantity
        law = term.law(gradients={q: face.normal_gradient(q) * face.normal})  # <- law-gradients
        return FaceFlux(dot(law, face.normal))

    _, _, u, flux, _ = _model()
    with pytest.raises(surface.RealizationMismatch) as caught:
        _numerics(u, flux, gradient_law())
    assert _at(caught.value, "law-gradients")
    assert "the law of a diffusive flux" in str(caught.value)
    with pytest.raises(surface.RealizationMismatch, match="the law of a diffusive flux"):
        _numerics(u, flux, two_point())


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


def test_a_speed_table_is_keyed_by_factors():
    """upwind(speed={factor: f}) (SD73): a key that is not a factor of a product support is a
    Python error when pops.Numerics runs the scheme (SU5); an empty table states no speed."""
    _, _, u, flux, _ = _model(lambda u: 2 * u)
    with pytest.raises(TypeError, match="keys its table by the factors of a product support"):
        _numerics(u, flux, upwind(speed={u: lambda face: 1.0}))
    _numerics(u, flux, upwind(speed={}))


def test_a_stated_speed_is_a_plain_function_of_face():
    """The options of a word of the library are checked when pops.Numerics runs its body (SU5),
    as the options of a scheme of one's own."""
    _, _, u, flux, _ = _model()
    with pytest.raises(surface.NotAPopsOperation, match="plain function, passed uncalled"):
        _numerics(u, flux, upwind(speed=central()))
    with pytest.raises(TypeError, match="takes a function of face"):
        _numerics(u, flux, upwind(speed=1.0))


def test_a_library_reconstruction_is_a_cell_scheme():
    _, _, u, flux, _ = _model()
    with pytest.raises(TypeError, match="takes a scheme that returns Reconstruct"):
        _numerics(u, flux, rusanov(reconstruction=two_point()))


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


# --- R1 of the review of #760: the average of a cell under a reconstruction -----------------------

def test_the_average_of_a_cell_under_a_reconstruction_names_its_lot():
    @surface.scheme
    def centre(cell, term, axis):
        return Reconstruct(cell.value(term.quantity))

    @surface.scheme
    def cell_average(face, term, reconstruction=None):
        q = term.quantity
        return FaceFlux(face.left_cell.value(q), reconstruction=reconstruction)

    _, _, u, flux, _ = _model()
    with pytest.raises(surface.NotYetElaborated, match="stencil faces of N7a"):
        _numerics(u, flux, cell_average(reconstruction=centre()))
    _numerics(u, flux, cell_average())


def test_a_later_kind_names_its_lot():
    from pops.surface.discrete import kinds

    with pytest.raises(surface.NotYetElaborated, match="face.gradient"):
        kinds.Gradient  # noqa: B018
    assert kinds.Flux is kinds.Flux


def test_a_stated_speed_that_reads_a_cell_under_a_reconstruction_names_its_lot():
    """U2 of the review of #760 at bdc25ca2: upwind(speed=f) whose f reads the average of a cell
    under a reconstruction is refused like a face scheme that does (R1)."""
    @surface.scheme
    def centre(cell, term, axis):
        return Reconstruct(cell.value(term.quantity))

    _, _, u, flux, _ = _model()

    def speed(face):
        return face.left_cell.value(u)

    with pytest.raises(surface.NotYetElaborated, match="stencil faces of N7a"):
        _numerics(u, flux, upwind(speed=speed, reconstruction=centre()))
