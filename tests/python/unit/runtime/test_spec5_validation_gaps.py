"""Spec 5 EARLY-VALIDATION gaps (sec.7 / sec.8.6; criteria 11 / 31; epic ADC-479).

Three validations the spec requires BEFORE runtime, each implemented under the OVERRIDING
no-false-positive discipline (only reject a KNOWN / declared incompatibility; a valid problem
must still pass):

* GAP 1 (sec.7, criterion 11): a :class:`pops.fields.FieldProblem` cross-checks the chosen
  elliptic solver's declared capabilities against the problem kind. A screened / anisotropic
  Poisson paired with a solver that declares ``supports_screened`` / ``supports_anisotropic``
  KNOWN-False (the spectral ``FFT``) is refused; ``GeometricMG`` (variable-epsilon) and a
  capability-less solver pass.
* GAP 2 (sec.7, criterion 11): a reconstruction whose DECLARED ghost depth exceeds an EXPLICIT
  block halo is refused (WENO5 needs 3; an explicit depth-2 block is too thin). The native
  runtime grows the halo to match the scheme, so WENO5 with no explicit constraint -- and MUSCL
  at depth 2 -- pass.
* GAP 3 (sec.8.6, criterion 31): :class:`pops.mesh.amr.Refine` accepts only real Handle
  declarations; the Problem registry authenticates and canonicalises them before runtime.

Pure Python; needs only ``import pops`` (nothing computes on a grid). The validations read
descriptor metadata and run nothing.
"""

import pytest

pops = pytest.importorskip("pops")

from pops.math import div, grad, laplacian, unknown  # noqa: E402
from pops.ir.expr import Var  # noqa: E402
from pops.fields import (  # noqa: E402
    AnisotropicPoissonProblem, PoissonProblem, ScreenedPoissonProblem)
from pops.fields.coefficients import ScalarCoefficient  # noqa: E402
from pops.solvers.elliptic import FFT, GeometricMG  # noqa: E402
from pops.numerics.reconstruction import (  # noqa: E402
    FirstOrder, MUSCL, WENO5, required_ghost_depth, validate_ghost_depth)
from pops.runtime._bricks_scheme import FiniteVolume  # noqa: E402
from pops.mesh.amr import Refine, TagUnion  # noqa: E402
from pops.mesh.layouts import AMR  # noqa: E402
from pops.mesh import CartesianMesh  # noqa: E402
from pops.fields.bcs import Periodic, Dirichlet  # noqa: E402
from pops.model import (  # noqa: E402
    DeclarationIndex, Handle, MissingOwnershipError, OwnerKind, OwnerPath)


# ----------------------------------------------------------------------------------------
# GAP 1 -- FieldProblem incompatible-solver validation (sec.7, criterion 11)
# ----------------------------------------------------------------------------------------

def _screened_problem(solver):
    phi = unknown("phi")
    rho = Var("rho", "cons")
    return ScreenedPoissonProblem(
        unknown=phi, equation=(-laplacian(phi) + 0.5 * phi == rho), solver=solver)


def _anisotropic_problem(solver):
    phi = unknown("phi")
    rho = Var("rho", "cons")
    eps_field = Handle("eps", kind="field", owner=OwnerPath.shared("validation.coefficient"))
    eps = ScalarCoefficient(eps_field)
    return AnisotropicPoissonProblem(
        unknown=phi, equation=(-div(eps * grad(phi)) == rho), solver=solver)


def test_screened_with_fft_is_rejected_with_actionable_message():
    # The FFT solver declares supports_screened KNOWN-False -> refused before runtime.
    with pytest.raises(ValueError) as exc:
        _screened_problem(FFT()).validate()
    msg = str(exc.value)
    assert "does not support a screened operator" in msg
    assert "supports_screened is False" in msg
    assert "pops.solvers.elliptic.GeometricMG()" in msg


def test_anisotropic_with_fft_is_rejected_with_actionable_message():
    with pytest.raises(ValueError) as exc:
        _anisotropic_problem(FFT()).validate()
    msg = str(exc.value)
    assert "does not support an anisotropic operator" in msg
    assert "supports_anisotropic is False" in msg
    assert "pops.solvers.elliptic.GeometricMG()" in msg


def test_screened_with_geometric_mg_is_fine():
    # NO FALSE POSITIVE: GeometricMG declares supports_variable_epsilon -> it serves the
    # screened reaction term; supports_screened=False is not a real incompatibility.
    assert GeometricMG().capabilities().supports("screened") is False
    assert GeometricMG().capabilities().supports("variable_epsilon") is True
    assert _screened_problem(GeometricMG()).validate() is True


def test_anisotropic_with_geometric_mg_is_fine():
    assert _anisotropic_problem(GeometricMG()).validate() is True


def test_capabilityless_solver_is_not_rejected():
    # NO FALSE POSITIVE: a solver that exposes no capabilities() dict (a bare object, an
    # external brick) has an ABSENT capability, not a declared-False one -> never rejected.
    assert _screened_problem(object()).validate() is True
    assert _anisotropic_problem(object()).validate() is True


def test_plain_poisson_with_fft_is_not_rejected():
    # NO FALSE POSITIVE: a plain Poisson needs neither screened nor anisotropic, so the FFT
    # solver (a real periodic constant-coefficient route) is not refused by this cross-check.
    phi = unknown("phi")
    rho = Var("rho", "cons")
    prob = PoissonProblem(unknown=phi, equation=(-laplacian(phi) == rho), solver=FFT())
    assert prob.validate() is True


def test_fft_with_non_periodic_bc_is_rejected():
    # Spec 5 #11: the FFT solver is periodic-only (supports_wall_bc False); pairing it with a
    # Dirichlet (wall) boundary is refused before runtime, naming the periodic / GeometricMG fix.
    phi = unknown("phi")
    rho = Var("rho", "cons")
    prob = PoissonProblem(unknown=phi, equation=(-laplacian(phi) == rho),
                          bcs=(Dirichlet(),), solver=FFT())
    with pytest.raises(ValueError) as exc:
        prob.validate()
    msg = str(exc.value)
    assert "requires a periodic boundary" in msg
    assert "supports_wall_bc is False" in msg
    assert "GeometricMG()" in msg


def test_fft_with_periodic_bc_is_fine():
    # NO FALSE POSITIVE: FFT + an explicitly periodic boundary is exactly its supported route.
    phi = unknown("phi")
    rho = Var("rho", "cons")
    prob = PoissonProblem(unknown=phi, equation=(-laplacian(phi) == rho),
                          bcs=(Periodic(),), solver=FFT())
    assert prob.validate() is True


def test_geometric_mg_with_dirichlet_is_fine():
    # NO FALSE POSITIVE: GeometricMG declares supports_wall_bc True -> it serves a Dirichlet wall.
    assert GeometricMG().capabilities().supports("wall_bc") is True
    phi = unknown("phi")
    rho = Var("rho", "cons")
    prob = PoissonProblem(unknown=phi, equation=(-laplacian(phi) == rho),
                          bcs=(Dirichlet(),), solver=GeometricMG())
    assert prob.validate() is True


# ----------------------------------------------------------------------------------------
# GAP 2 -- WENO5 ghost-depth insufficiency (sec.7, criterion 11)
# ----------------------------------------------------------------------------------------

def test_declared_required_ghost_depths():
    # The declared per-scheme requirement: WENO5 >= 3, MUSCL >= 2, first-order 1.
    assert required_ghost_depth(WENO5()) == 3
    assert required_ghost_depth(MUSCL()) == 2
    assert required_ghost_depth(FirstOrder()) == 1
    assert required_ghost_depth("weno5") == 3
    assert required_ghost_depth("minmod") == 2


def test_weno5_on_explicit_depth2_block_is_rejected():
    # An EXPLICIT depth-2 block is too thin for WENO5's 3-cell stencil -> clear error.
    with pytest.raises(ValueError) as exc:
        FiniteVolume(weno5=True).validate(ghost_depth=2, block="plasma")
    msg = str(exc.value)
    assert "WENO5 requires ghost_depth >= 3" in msg
    assert "block 'plasma' has ghost_depth=2" in msg
    # The same check fires on the raw token form too.
    with pytest.raises(ValueError, match="WENO5 requires ghost_depth >= 3"):
        validate_ghost_depth("weno5", available=2, block="plasma")


def test_muscl_on_depth2_block_is_fine():
    # NO FALSE POSITIVE: a second-order MUSCL scheme fits the default 2-cell halo.
    assert FiniteVolume(minmod=True).validate(ghost_depth=2) is True
    assert validate_ghost_depth("minmod", available=2) is True


def test_weno5_without_explicit_constraint_is_not_rejected():
    # NO FALSE POSITIVE: the native runtime grows the block halo to match the scheme
    # (block_n_ghost("weno5") == 3), so WENO5 on a default block is a VALID problem.
    assert FiniteVolume(weno5=True).validate() is True
    assert validate_ghost_depth(WENO5()) is True
    assert validate_ghost_depth("weno5") is True


def test_undeclared_reconstruction_is_not_rejected():
    # NO FALSE POSITIVE: an unknown token has no declared requirement -> never rejected.
    assert validate_ghost_depth("custom_user_scheme", available=1) is True


# ----------------------------------------------------------------------------------------
# GAP 3 -- typed Refine references + registry authentication (sec.8.6, criterion 31)
# ----------------------------------------------------------------------------------------

class _FakeModel:
    """Minimal model with an authoritative declaration index."""

    def __init__(self):
        self.name = "validation-model"
        self.owner_path = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, self.name)
        self.rho = Handle("rho", kind="state", owner=self.owner_path)
        self.density = Handle("Density", kind="role", owner=self.owner_path)
        self.b_z = Handle("B_z", kind="aux", owner=self.owner_path)

    def declaration_index(self):
        return DeclarationIndex(
            owner=self.owner_path, handles=(self.rho, self.density, self.b_z))


def test_refine_rejects_flat_names_at_construction():
    with pytest.raises(TypeError, match="names and strings"):
        Refine.on("rho")


def test_refine_on_real_handles_self_validates_its_shape():
    model = _FakeModel()
    assert Refine.on(model.rho).above(0.05).validate() is True
    assert Refine.on(model.density).above(0.05).validate() is True
    assert Refine.on(model.b_z).above(0.05).validate() is True
    with pytest.raises(ValueError, match="incomplete"):
        Refine.on(model.rho).validate()


def test_problem_amr_refine_authenticates_and_canonicalizes_the_handle():
    model = _FakeModel()
    prob = pops.Problem(layout=AMR(base=CartesianMesh(n=64))).block("plasma", physics=model)
    union = TagUnion(
        Refine.on(model.rho).above(0.05),
        Refine.on(model.density).gradient_above(0.5),
    )
    assert prob.amr.refine(union) is prob
    stored = prob._constraints.refinement["refine"]
    assert all(item.subject.is_resolved for item in stored.criteria)


def test_problem_amr_refine_rejects_same_owner_handle_missing_from_index():
    model = _FakeModel()
    prob = pops.Problem(layout=AMR(base=CartesianMesh(n=64))).block("plasma", physics=model)
    bogus = Handle("definitely_not_a_role", kind="role", owner=model.owner_path)
    with pytest.raises(MissingOwnershipError, match="not registered"):
        prob.amr.refine(Refine.on(bogus).above(0.05))
