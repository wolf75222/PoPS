"""Spec 5 (sec.9 / sec.5.5 / sec.14.2.4): the pops.fields + pops.numerics.terms surface.

These exercise the inert authoring descriptors of an elliptic field solve: the
:class:`FieldProblem` / :class:`PoissonProblem` declarations built from a real
``pops.math.Equation``, the validation that rejects a missing solver or a non-Equation,
the typed bcs / rhs / coefficients / nullspace / aux pieces, and the typed right-hand-side
composition terms (Flux / SourceTerm / LocalTerm). Pure Python; needs only ``import pops``
(nothing here computes on a grid).
"""

import pytest

pops = pytest.importorskip("pops")

from pops.math import laplacian, unknown  # noqa: E402
from pops.ir.expr import Var  # noqa: E402  (the board source-expression node: Var("rho", "cons"))
from pops.fields import (  # noqa: E402
    FieldProblem, PoissonProblem, ScreenedPoissonProblem, AnisotropicPoissonProblem)
from pops.fields import bcs, rhs, coefficients, nullspace, aux  # noqa: E402
from pops.model import Handle, Module, OwnerPath  # noqa: E402
from pops.numerics.terms import DefaultSource, Flux, SourceTerm, LocalTerm  # noqa: E402
from pops.physics._facade import Model as PhysicsModel  # noqa: E402
from pops.problem import Case  # noqa: E402


def _shared_field(name):
    return Handle(name, kind="field", owner=OwnerPath.shared("mesh.fields_authoring"))


def _registered_blocks(*names):
    problem = Case(name="field-authoring-case")
    module = Module("charge-model")
    return problem, tuple(problem.block(name, module) for name in names)


def _source_operator(name="ionization"):
    model = PhysicsModel("field-authoring-%s" % name)
    (u,) = model.conservative_vars("u")
    return model.source_term(name, [-u])


def _poisson_equation():
    # Build -laplacian(phi) == alpha*(rho - rho_ref) from real PoPS operators: the unknown
    # via pops.math.unknown, the source fields as board Var nodes (the same idiom physics
    # sources use, e.g. Var("ni","cons") - Var("ne","cons")).
    phi = unknown("phi")
    rho = Var("rho", "cons")
    rho_ref = Var("rho_ref", "aux")
    alpha = 2.0
    return phi, (-laplacian(phi) == alpha * (rho - rho_ref))


def test_package_exports():
    assert "fields" in pops.__all__
    assert pops.fields is not None
    # Spec 5 criterion 7: pops.lib is presets-only -- the elliptic-field brick catalog moved out
    # of pops.lib.fields. The typed authoring package (pops.fields) is DISTINCT from its own brick
    # catalog (pops.fields.catalog), and pops.lib no longer exposes a fields catalog at all.
    assert pops.fields is not pops.fields.catalog
    assert not hasattr(pops.lib, "fields")


def test_poisson_problem_stores_equation():
    from pops.math import Equation

    phi, eq = _poisson_equation()
    prob = PoissonProblem(unknown=phi, equation=eq, solver=object())
    assert prob.equation is eq
    assert isinstance(prob.equation, Equation)
    assert prob.capabilities().to_dict()["poisson"] is True


def test_validate_passes_with_solver():
    phi, eq = _poisson_equation()
    prob = PoissonProblem(unknown=phi, equation=eq, solver=object())
    assert prob.validate() is True
    assert prob.available().ok


def test_validate_requires_solver():
    phi, eq = _poisson_equation()
    prob = PoissonProblem(unknown=phi, equation=eq, solver=None)
    with pytest.raises(ValueError):
        prob.validate()
    assert not prob.available()


def test_validate_rejects_bool_equation():
    # The common mistake: '==' on plain values yields a Python bool, not an Equation.
    prob = FieldProblem(equation=(1 == 1), solver=object())
    with pytest.raises(TypeError):
        prob.validate()


def test_validate_rejects_non_equation():
    prob = FieldProblem(equation="-laplacian(phi) == rhs", solver=object())
    with pytest.raises(TypeError):
        prob.validate()


def test_poisson_rejects_non_laplacian_lhs():
    phi = unknown("phi")
    # A first-derivative LHS is not an elliptic Poisson operator.
    from pops.math import dx

    prob = PoissonProblem(unknown=phi, equation=(dx(phi) == 0.0), solver=object())
    with pytest.raises(ValueError):
        prob.validate()


def test_poisson_subclasses_validate_their_forms():
    # ADC-491: the Poisson-family shortcuts now validate their DISTINGUISHING elliptic form
    # (constructible since the board-node elliptic algebra landed), not the plain laplacian.
    from pops.math import div, grad
    from pops.fields.coefficients import ScalarCoefficient

    phi = unknown("phi")
    rho = Var("rho", "cons")
    screened = ScreenedPoissonProblem(
        unknown=phi, equation=(-laplacian(phi) + 0.5 * phi == rho), solver=object())
    assert isinstance(screened, FieldProblem)
    assert screened.validate() is True

    eps = ScalarCoefficient(_shared_field("eps"))
    aniso = AnisotropicPoissonProblem(
        unknown=phi, equation=(-div(eps * grad(phi)) == rho), solver=object())
    assert isinstance(aniso, FieldProblem)
    assert aniso.validate() is True


def test_bcs_construct_and_inspect():
    for cond in (bcs.Periodic(), bcs.Dirichlet(value=1.5), bcs.Neumann(value=0.0),
                 bcs.FirstOrderExtrapolation()):
        assert cond.category == "field_bc"
        assert isinstance(cond.inspect(), dict)
    fb = bcs.FaceBC(bcs.XMin(), bcs.Dirichlet(value=0.0))
    assert fb.options()["face"] == "XMin"


def test_facebc_rejects_non_face():
    with pytest.raises(TypeError):
        bcs.FaceBC("xmin", bcs.Dirichlet())
    with pytest.raises(TypeError):
        bcs.FaceBC(bcs.Dirichlet(), bcs.Dirichlet())  # condition is not a face


def test_rhs_from_blocks():
    problem, (ions, electrons) = _registered_blocks("ions", "electrons")
    expected = [problem.resolve(block).qualified_id for block in (ions, electrons)]
    cd = rhs.ChargeDensity.from_blocks(ions, electrons)
    resolved = cd.resolve_references(problem.resolve)
    assert resolved.options()["blocks"] == expected
    assert resolved.requirements().to_dict()["blocks"] == expected
    assert cd.blocks == (ions, electrons)
    # Single-iterable form is also accepted.
    cd2 = rhs.ChargeDensity.from_blocks([ions, electrons])
    assert cd2.resolve_references(problem.resolve).options()["blocks"] == expected


def test_reference_descriptors_reject_strings():
    with pytest.raises(TypeError, match="BlockHandle"):
        rhs.ChargeDensity.from_blocks("ions")
    with pytest.raises(TypeError, match="declaration Handle"):
        rhs.FixedSource("rho_background")
    with pytest.raises(TypeError, match="declaration Handle"):
        coefficients.ScalarCoefficient("eps")
    with pytest.raises(TypeError, match="typed OperatorHandle"):
        SourceTerm("ionization")
    with pytest.raises(TypeError, match="typed OperatorHandle"):
        LocalTerm("ionization")


def test_coefficients_and_nullspace():
    eps = _shared_field("eps")
    reaction = _shared_field("k")
    sc = coefficients.ScalarCoefficient(eps)
    rc = coefficients.ReactionCoefficient(reaction)
    assert sc.requirements().to_dict()["aux_field"] == eps.qualified_id
    assert rc.options()["role"] == "reaction"
    ns = nullspace.ConstantNullspace()
    assert ns.capabilities().to_dict()["removes_constant"] is True


def test_aux_static_derived_and_halo():
    sa = aux.StaticAux("eps0", value=8.85e-12)
    da = aux.DerivedAux("E", expression=None)
    assert sa.options()["kind"] == "static"
    assert da.options()["kind"] == "derived"
    # AuxHalo is re-exported from pops.mesh.aux (same descriptor, not a copy).
    from pops.mesh.aux import AuxHalo as MeshAuxHalo

    assert aux.AuxHalo is MeshAuxHalo


def test_numerics_terms_construct_and_options():
    source = _source_operator()
    flux = Flux()
    default = DefaultSource()
    src = SourceTerm(source)
    loc = LocalTerm(source)
    assert flux.options()["term"] == "flux"
    assert default.options()["term"] == "default_source"
    assert src.operator is source
    assert src.options()["operator"]["local_id"] == "ionization"
    assert loc.options()["term"] == "local"
    assert loc.operator is source
    assert flux.capabilities().to_dict()["conservative"] is True


def test_print_summaries_are_short_and_named():
    phi, eq = _poisson_equation()
    prob = PoissonProblem(unknown=phi, equation=eq, solver=None)
    # The class-named summaries: problem + the rhs composition terms + the type-named bcs.
    class_named = [prob, Flux(), DefaultSource(), bcs.Dirichlet(),
                   nullspace.ConstantNullspace()]
    for obj in class_named:
        text = str(obj)
        assert len(text) < 300, "summary too long for %s: %r" % (type(obj).__name__, text)
        assert text.startswith(type(obj).__name__), text
    source = _source_operator("reaction")
    for obj in (SourceTerm(source), LocalTerm(source)):
        text = str(obj)
        # A named term exposes the complete typed operator identity, so its transparent
        # summary is intentionally richer than the class-only descriptor summaries above.
        assert len(text) < 600, "summary too long for %s: %r" % (type(obj).__name__, text)
        assert text.startswith(source.name), text
        assert source.inspect()["qualified_id"] in text
        assert "#authoring=" not in text
    # Instance-named descriptors lead with their field name (the default Descriptor head),
    # and stay short too.
    problem, (ions,) = _registered_blocks("ions")
    for obj in (rhs.ChargeDensity.from_blocks(ions).resolve_references(problem.resolve),
                coefficients.ScalarCoefficient(_shared_field("eps")), aux.StaticAux("eps0")):
        text = str(obj)
        assert len(text) < 300, "summary too long for %s: %r" % (type(obj).__name__, text)
        assert text.startswith(obj.name), text
