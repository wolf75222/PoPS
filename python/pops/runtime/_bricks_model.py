"""Model bricks : state / transport / source / elliptic value objects (Spec-4 PR-F).

The composable bricks a MODEL is built from, plus the ``Model`` composer (ModelSpec) and the elliptic physical
model (EPM) bricks/helpers. ``pops.runtime.bricks`` re-exports everything here together with the
scheme/time policies in ``_bricks_scheme``. ``ModelSpec`` comes from the loaded extension via
``pops._bootstrap``.
"""

from __future__ import annotations

from typing import Any

from pops._bootstrap import ModelSpec
from pops.runtime._numeric import exact_real, native_real
from pops.runtime.defaults import (
    PHYSICAL_DEFAULT_ALPHA,
    PHYSICAL_DEFAULT_B0,
    PHYSICAL_DEFAULT_BACKGROUND_N0,
    PHYSICAL_DEFAULT_CHARGE_Q,
    PHYSICAL_DEFAULT_FLUID_STATE_CS2,
    PHYSICAL_DEFAULT_FOUR_PI_G,
    PHYSICAL_DEFAULT_GAMMA,
    PHYSICAL_DEFAULT_GRAVITY_RHO0,
    PHYSICAL_DEFAULT_GRAVITY_SIGN,
    PHYSICAL_DEFAULT_NATIVE_ISOTHERMAL_CS2,
    PHYSICAL_DEFAULT_QOM,
    PHYSICAL_DEFAULT_VACUUM_FLOOR,
)


# --- State bricks ---------------------------------------------------------
class Scalar:
    """Scalar state (1 variable, e.g. a transported density)."""


class FluidState:
    """Fluid state. kind = "compressible" (gamma) or "isothermal" (cs2).

    vacuum_floor (isothermal only, ADC-77): quasi-vacuum density floor. When > 0 the model computes
    the velocity as u = m/max(rho, vacuum_floor), bounding the wave speed and the advective flux where
    the flow evacuates the background (rho -> ~0). It does NOT modify the conserved state (only the
    velocity estimate). 0 (default) = inactive (bit-identical). This is independent of the spatial
    positivity_floor (the Zhang-Shu reconstruction limiter): the two address different failure modes
    and must be enabled separately. It is carried by the native ``ModelSpec`` route; generated
    production packages carry their own immutable transport parameters.
    """

    def __init__(self,
                 kind: str = "compressible",
                 gamma: Any = PHYSICAL_DEFAULT_GAMMA,
                 cs2: Any = PHYSICAL_DEFAULT_FLUID_STATE_CS2,
                 vacuum_floor: Any = PHYSICAL_DEFAULT_VACUUM_FLOOR) -> None:
        self.kind = kind
        self.gamma = exact_real(gamma, where="FluidState.gamma")
        self.cs2 = exact_real(cs2, where="FluidState.cs2")
        self.vacuum_floor = exact_real(
            vacuum_floor, where="FluidState.vacuum_floor", minimum=0)

    @classmethod
    def compressible(cls, gamma: Any = PHYSICAL_DEFAULT_GAMMA) -> Any:
        """Typed constructor for the COMPRESSIBLE fluid state (Spec 5 sec.14.2.5).

        ``pops.FluidState.compressible(gamma=1.4)`` is the typed equivalent of
        ``pops.FluidState(kind="compressible", gamma=1.4)``: it builds the SAME inert state object
        (kind="compressible", carrying gamma -> spec.gamma via Model) instead of selecting the kind
        with a magic string. Pairs with CompressibleFlux (4 variables [rho, rho_u, rho_v, E]).
        """
        return cls(kind="compressible", gamma=gamma)

    @classmethod
    def isothermal(cls,
                   cs2: Any = PHYSICAL_DEFAULT_FLUID_STATE_CS2,
                   vacuum_floor: Any = PHYSICAL_DEFAULT_VACUUM_FLOOR) -> Any:
        """Typed constructor for the ISOTHERMAL fluid state (Spec 5 sec.14.2.5).

        ``pops.FluidState.isothermal(cs2=0.5, vacuum_floor=0.0)`` is the typed equivalent of
        ``pops.FluidState(kind="isothermal", cs2=0.5, vacuum_floor=0.0)``: it builds the SAME inert
        state object (kind="isothermal", carrying cs2 -> spec.cs2 and vacuum_floor ->
        spec.vacuum_floor via Model). Pairs with IsothermalFlux (3 variables [rho, rho_u, rho_v]).
        See the class docstring for the vacuum_floor (ADC-77) semantics.
        """
        return cls(kind="isothermal", cs2=cs2, vacuum_floor=vacuum_floor)


# --- Transport bricks ---------------------------------------------------
class ExB:
    """Scalar advection by the E x B drift (magnetic field B0)."""

    def __init__(self, B0: Any = PHYSICAL_DEFAULT_B0) -> None:
        self.B0 = exact_real(B0, where="ExB.B0")


class CompressibleFlux:
    """Compressible Euler flux (gamma comes from the FluidState state)."""


class IsothermalFlux:
    """Isothermal Euler flux for the native ``ModelSpec`` route.

    ``cs2`` and ``vacuum_floor`` are immutable descriptor values. Generated models use their
    authenticated BindSchema vector instead of a second mutable parameter channel.
    """

    def __init__(self, cs2: Any = PHYSICAL_DEFAULT_NATIVE_ISOTHERMAL_CS2,
                 vacuum_floor: Any = PHYSICAL_DEFAULT_VACUUM_FLOOR) -> None:
        self.cs2 = exact_real(cs2, where="IsothermalFlux.cs2")
        self.vacuum_floor = exact_real(
            vacuum_floor, where="IsothermalFlux.vacuum_floor", minimum=0)


# --- Source bricks ------------------------------------------------------
class NoSource:
    """No source."""


class PotentialForce:
    """Potential force (q/m) rho E on the momentum (+ work if 4 vars)."""

    def __init__(self, charge: Any = PHYSICAL_DEFAULT_QOM) -> None:
        self.charge = exact_real(charge, where="PotentialForce.charge")


class GravityForce:
    """Gravitational force rho g (+ work if 4 vars)."""


class MagneticLorentzForce:
    """MAGNETIC Lorentz force q (v x B_z) on the momentum (native C++ brick
    pops::MagneticLorentzForce, exposed to the Python API by the 2026-06 audit).

    EXPLICIT regime (moderate omega_c): pointwise algebraic term, no work (F . v = 0, energy
    unchanged). Reads B_z from the aux channel (canonical component 3): call
    ``sim.set_magnetic_field(Bz)`` to populate it. Requires a fluid transport >= 3 variables (momentum
    on 2 axes); rejected on a scalar. The STIFF regime (large omega_c) is authored as an explicit
    condensed ``Program.solve`` graph, NOT through this pointwise brick.

    ``charge`` = q/m, sign included (same convention as PotentialForce)."""

    def __init__(self, charge: Any = PHYSICAL_DEFAULT_QOM) -> None:
        self.charge = exact_real(charge, where="MagneticLorentzForce.charge")


class PotentialMagneticForce:
    """Electrostatic force + magnetic Lorentz SUMMED: (q/m) rho E + q (v x B_z) (native C++
    brick CompositeSource<PotentialForce, MagneticLorentzForce>, the full magnetized diocotron
    force). Same q/m for both forces (same species). Reads B_z (set_magnetic_field); requires a
    fluid transport >= 3 variables. ``charge`` = q/m, sign included."""

    def __init__(self, charge: Any = PHYSICAL_DEFAULT_QOM) -> None:
        self.charge = exact_real(charge, where="PotentialMagneticForce.charge")


# --- Elliptic right-hand-side bricks ------------------------------------
class ChargeDensity:
    """Charge density f = q n."""

    def __init__(self, charge: Any = PHYSICAL_DEFAULT_CHARGE_Q) -> None:
        self.charge = exact_real(charge, where="ChargeDensity.charge")


class BackgroundDensity:
    """Neutralizing background f = alpha (n - n0)."""

    def __init__(self,
                 alpha: Any = PHYSICAL_DEFAULT_ALPHA,
                 n0: Any = PHYSICAL_DEFAULT_BACKGROUND_N0) -> None:
        self.alpha = exact_real(alpha, where="BackgroundDensity.alpha")
        self.n0 = exact_real(n0, where="BackgroundDensity.n0")


class GravityCoupling:
    """Self-consistent coupling f = sign 4piG (rho - rho0). sign = +1 gravity, -1 plasma."""

    def __init__(self,
                 sign: Any = PHYSICAL_DEFAULT_GRAVITY_SIGN,
                 four_pi_G: Any = PHYSICAL_DEFAULT_FOUR_PI_G,
                 rho0: Any = PHYSICAL_DEFAULT_GRAVITY_RHO0) -> None:
        self.sign = exact_real(sign, where="GravityCoupling.sign")
        self.four_pi_G = exact_real(four_pi_G, where="GravityCoupling.four_pi_G")
        self.rho0 = exact_real(rho0, where="GravityCoupling.rho0")


def Model(state: Any, transport: Any, source: Any, elliptic: Any) -> Any:
    """Compose a model (ModelSpec) from state, transport, source, elliptic bricks.

    Validates the state <-> transport consistency (Scalar with ExB; compressible FluidState with
    CompressibleFlux; isothermal with IsothermalFlux) and carries the parameters into the spec.

    The returned ``ModelSpec`` is the BOUNDED LEGACY BRIDGE for the native ``add_block`` path (a
    flat C++ POD of brick tags + parameters); it is NOT the target representation. The target
    representation of a model is the operator-first ``pops.model.Module`` (compiled to a Problem)
    and its self-describing ``ModuleManifest`` (ADC-585). ADC-585 also moved this POD off the pops
    root: it lives at ``pops.runtime.ModelSpec``, not ``pops.ModelSpec``.
    """
    spec: Any = ModelSpec()

    if isinstance(state, Scalar):
        if not isinstance(transport, ExB):
            raise ValueError("Scalar requires transport=ExB(...)")
    elif isinstance(state, FluidState):
        if state.kind == "compressible":
            spec.gamma = native_real(state.gamma, where="Model.gamma")
            if not isinstance(transport, CompressibleFlux):
                raise ValueError("FluidState(compressible) requires transport=CompressibleFlux()")
        elif state.kind == "isothermal":
            spec.cs2 = native_real(state.cs2, where="Model.cs2")
            spec.vacuum_floor = native_real(
                getattr(state, "vacuum_floor", 0.0), where="Model.vacuum_floor")
            if not isinstance(transport, IsothermalFlux):
                raise ValueError("FluidState(isothermal) requires transport=IsothermalFlux()")
        else:
            raise ValueError("FluidState.kind: 'compressible' | 'isothermal'")
    else:
        raise ValueError("state: pops.Scalar() | pops.FluidState(...)")

    if isinstance(transport, ExB):
        spec.transport = "exb"; spec.B0 = native_real(transport.B0, where="Model.B0")
    elif isinstance(transport, CompressibleFlux):
        spec.transport = "compressible"
    elif isinstance(transport, IsothermalFlux):
        spec.transport = "isothermal"
    else:
        raise ValueError("transport: ExB | CompressibleFlux | IsothermalFlux")

    if isinstance(source, NoSource):
        spec.source = "none"
    elif isinstance(source, PotentialForce):
        spec.source = "potential"; spec.qom = native_real(source.charge, where="Model.qom")
    elif isinstance(source, GravityForce):
        spec.source = "gravity"
    elif isinstance(source, MagneticLorentzForce):
        spec.source = "magnetic"; spec.qom = native_real(source.charge, where="Model.qom")
    elif isinstance(source, PotentialMagneticForce):
        spec.source = "potential_magnetic"; spec.qom = native_real(source.charge, where="Model.qom")
    else:
        raise ValueError("source: NoSource | PotentialForce | GravityForce | MagneticLorentzForce "
                         "| PotentialMagneticForce")

    if isinstance(elliptic, ChargeDensity):
        spec.elliptic = "charge"; spec.q = native_real(elliptic.charge, where="Model.q")
    elif isinstance(elliptic, BackgroundDensity):
        spec.elliptic = "background"
        spec.alpha = native_real(elliptic.alpha, where="Model.alpha")
        spec.n0 = native_real(elliptic.n0, where="Model.n0")
    elif isinstance(elliptic, GravityCoupling):
        spec.elliptic = "gravity"
        spec.sign = native_real(elliptic.sign, where="Model.sign")
        spec.four_pi_G = native_real(elliptic.four_pi_G, where="Model.four_pi_G")
        spec.rho0 = native_real(elliptic.rho0, where="Model.rho0")
    else:
        raise ValueError("elliptic: ChargeDensity | BackgroundDensity | GravityCoupling")

    return spec


# --- Elliptic model (EPM): Poisson = a composable instance ------------
# The elliptic model is not a hard-coded special case; it is an EllipticPhysicalModel
# composed of bricks (operator + right-hand side + output). Poisson is its current instance.
class DivEpsGrad:
    """Elliptic operator D = div(eps grad .). eps constant (1.0 = Poisson). Variable eps(x) and
    other operators (diffusion, projection) are refinements (they would touch the solver)."""

    def __init__(self, epsilon: Any = 1.0) -> None:
        self.epsilon = exact_real(epsilon, where="DivEpsGrad.epsilon")


class CompositeRhs:
    """System right-hand side f = sum_s elliptic_rhs_s(u_s): the SUM of the elliptic bricks
    carried by the blocks. Each block chooses its brick (charge q n, background alpha (n-n0), gravity
    coupling sign 4piG (rho-rho0)) via Model(elliptic=...); this right-hand side assembles them. It is the
    GENERIC right-hand side of the EPM: it assumes NO particular form for the contributions."""


class ChargeDensitySource(CompositeRhs):
    """Usual case of the composite right-hand side: all blocks carry a charge density, so
    f = sum_s q_s n_s. Historical alias of CompositeRhs (the computation stays the sum of the bricks)."""


class ElectricFieldFromPotential:
    """Post-processing: E = -grad phi, reinjected into aux of the hyperbolic models."""


class EllipticModel:
    """EllipticPhysicalModel: unknown + operator + right-hand side + output."""

    def __init__(self, unknown: Any, operator: Any, rhs: Any, output: Any) -> None:
        self.unknown = unknown
        self.operator = operator
        self.rhs = rhs
        self.output = output


def div_eps_grad(epsilon: Any = 1.0) -> Any:
    return DivEpsGrad(epsilon)


def charge_density() -> Any:
    return ChargeDensitySource()


def composite_rhs() -> Any:
    """Generic right-hand side f = sum_s elliptic_rhs_s(u_s) (sum of the per-block bricks)."""
    return CompositeRhs()


def electric_field_from_potential() -> Any:
    return ElectricFieldFromPotential()


def elliptic(unknown: Any = "phi", operator: Any = None, rhs: Any = None,
             output: Any = None) -> Any:
    """Compose an EPM. Poisson = elliptic(operator=div_eps_grad(), rhs=charge_density(),
    output=electric_field_from_potential()). The right-hand side can be composite_rhs() (GENERIC
    sum of the per-block elliptic bricks: charge, background, gravity); charge_density() is
    the usual case (alias)."""
    return EllipticModel(unknown, operator or DivEpsGrad(), rhs or CompositeRhs(),
                         output or ElectricFieldFromPotential())


class EllipticSolver:
    """Elliptic solver: 'geometric_mg' (any case, wall) | 'fft' (periodic, n = 2^k, discrete
    stencil) | 'fft_spectral' (periodic, continuous symbol -(kx^2+ky^2): fidelity to spectral
    references such as poisson_fft.m, exact on sinusoids)."""

    def __init__(self, kind: str = "geometric_mg") -> None:
        self.kind = kind
