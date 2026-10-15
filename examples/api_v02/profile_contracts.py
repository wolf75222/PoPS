"""Reviewable implementation contracts, separate from scientific qualification.

This examples-only catalogue is not a physics registry in the engine. A row is
a precisely chosen set of equations and compositions, with named compatibility
profiles. It contains neither an implementation-success flag nor native PASSes.

An implementation manifest binds each required component to exact Python source
bytes and a symbol. Checking that manifest verifies coverage and provenance; it
does not prove that arbitrary Python implements the stated mathematics. Native
execution, numerical experiments and mathematical claims remain separate evidence.
In particular a model's lack of hyperbolicity cannot be repaired by relabelling
it as a missing API operation, or by making require_complete demand a theorem.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from types import MappingProxyType


@dataclass(frozen=True)
class BuilderRef:
    symbol: str
    parameters: tuple[tuple[str, object], ...] = ()

    def kwargs(self):
        return dict(self.parameters)


@dataclass(frozen=True)
class NamedProfile:
    name: str
    builder: BuilderRef
    scope: str


@dataclass(frozen=True)
class RequiredComponent:
    identifier: str
    description: str
    parameter_choices: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class ComponentBinding:
    component: str
    source_symbol: str
    source_sha256: str


@dataclass(frozen=True)
class ImplementationManifest:
    profile: str
    bindings: tuple[ComponentBinding, ...]
    dependencies: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ProfileSelection:
    """The actual example selected for construction, not an execution receipt."""

    identifier: str
    name: str
    builder: str
    parameters: tuple[tuple[str, object], ...]
    parameter_overrides: tuple[str, ...]
    implementation_manifest: ImplementationManifest | None = None

    def kwargs(self):
        return dict(self.parameters)


class MissingComponentsError(ValueError):
    def __init__(self, profile, missing):
        self.profile, self.missing = profile, tuple(sorted(missing))
        super().__init__(
            profile + " lacks required equation/composition components: " + ", ".join(self.missing)
        )


@dataclass(frozen=True)
class ProfileContract:
    identifier: str
    variant: str
    builder: BuilderRef
    required_components: tuple[RequiredComponent, ...]
    assumptions: tuple[str, ...]
    unestablished_properties: tuple[str, ...]
    numerical_receivers: tuple[str, ...]
    compatibility_profiles: tuple[NamedProfile, ...] = ()
    companion_profiles: tuple[NamedProfile, ...] = ()
    model_properties: tuple[str, ...] = ()

    @property
    def key(self):
        return self.identifier + ":" + self.variant

    def missing_components(self, manifest, *, parameters=None):
        if manifest.profile != self.key:
            raise ValueError("implementation manifest belongs to a different selected profile")
        names = tuple(binding.component for binding in manifest.bindings)
        if len(set(names)) != len(names):
            raise ValueError("a required component must have one unambiguous binding")
        required = {component.identifier for component in self.required_components}
        unexpected = set(names) - required
        if unexpected:
            raise ValueError("unknown component bindings: " + ", ".join(sorted(unexpected)))
        missing = required - set(names)
        if parameters is not None:
            missing.update(
                component.identifier
                for component in self.required_components
                if any(parameters.get(name) != value for name, value in component.parameter_choices)
            )
        return tuple(sorted(missing))

    def require_components(self, manifest, *, source_root, parameters=None):
        """Require all components, then authenticate their actual source bytes.

        No receipt, convergence claim, regime restriction or theorem participates
        in this coverage test. Missing equations and stale source fail explicitly.
        A caller still must obtain native and numerical evidence separately.
        """
        missing = self.missing_components(manifest, parameters=parameters)
        if missing:
            raise MissingComponentsError(self.key, missing)
        root = Path(source_root).resolve()

        def authenticate(module, expected):
            if not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", module):
                raise ValueError("component source must be an explicit Python module:symbol")
            path = root.joinpath(*module.split(".")).with_suffix(".py").resolve()
            if not path.is_relative_to(root):
                raise ValueError("component source escapes its declared source tree")
            source = path.read_bytes()
            if sha256(source).hexdigest() != expected:
                raise ValueError("component source digest changed: " + module)
            return source

        for binding in manifest.bindings:
            module, separator, symbol = binding.source_symbol.partition(":")
            if not separator:
                raise ValueError("component source must be an explicit Python module:symbol")
            source = authenticate(module, binding.source_sha256)
            definitions = {
                node.name
                for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            if symbol not in definitions:
                raise ValueError("component source symbol is absent: " + binding.source_symbol)
        for module, expected in manifest.dependencies:
            authenticate(module, expected)
        return manifest


@dataclass(frozen=True)
class NumericalReceipt:
    """An explicit external attestation, never synthesized by the catalogue.

    A receipt applies only to its variant, source, artifact, parameters, backend
    and ranks. Merely naming a test file in numerical_receivers is not a receipt.
    The caller must read/verify the referenced evidence before constructing one.
    """

    profile: str
    source_sha256: str
    artifact_sha256: str
    parameters_sha256: str
    backend: str
    ranks: int
    evidence: str
    checks: tuple[str, ...]
    outcome: str

    def __post_init__(self):
        if self.outcome not in ("pass", "fail", "not_run"):
            raise ValueError("receipt outcome is pass, fail or not_run")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in (self.source_sha256, self.artifact_sha256, self.parameters_sha256)
        ):
            raise ValueError(
                "a numerical receipt needs exact source/artifact/parameter SHA256 identities"
            )
        if type(self.ranks) is not int or self.ranks < 1 or not self.backend or not self.evidence:
            raise ValueError(
                "a numerical receipt needs a backend, positive MPI size and external evidence"
            )
        if self.outcome == "pass" and not self.checks:
            raise ValueError("a numerical PASS must name the checks that actually passed")


def matching_receipts(
    contract, receipts, *, source_sha256, artifact_sha256, parameters_sha256, backend, ranks
):
    """Return only exact-identity evidence; failures remain visible as failures."""
    return tuple(
        receipt
        for receipt in receipts
        if (
            receipt.profile,
            receipt.source_sha256,
            receipt.artifact_sha256,
            receipt.parameters_sha256,
            receipt.backend,
            receipt.ranks,
        )
        == (contract.key, source_sha256, artifact_sha256, parameters_sha256, backend, ranks)
    )


_PREFIX = "examples.api_v02."
_TEST = "tests/python/integration/runtime/"


def _builder(symbol, **parameters):
    return BuilderRef(_PREFIX + symbol, tuple(parameters.items()))


def _named(name, symbol, scope, **parameters):
    return NamedProfile(name, _builder(symbol, **parameters), scope)


def _p(
    identifier,
    variant,
    symbol,
    components,
    assumptions,
    unestablished,
    receivers,
    *,
    parameters=None,
    compatibility=(),
    companions=(),
    model_properties=(),
    component_choices=None,
):
    required = tuple(RequiredComponent(*item.split(":", 1)) for item in components)
    choices = component_choices or {}
    required = tuple(
        RequiredComponent(
            item.identifier, item.description, tuple(choices.get(item.identifier, {}).items())
        )
        for item in required
    )
    if len({item.identifier for item in required}) != len(required) or not required:
        raise ValueError("each profile needs distinct nonempty equation/component obligations")
    return ProfileContract(
        identifier,
        variant,
        _builder(symbol, **(parameters or {})),
        required,
        tuple(assumptions),
        tuple(unestablished),
        tuple(_TEST + name for name in receivers),
        tuple(compatibility),
        tuple(companions),
        tuple(model_properties),
    )


_MODELS = (
    _p(
        "M01",
        "periodic-advection",
        "models_M01_M14:build_m01",
        (
            "transport:Authored conservative advection flux",
            "face-incidence:Shared face deliveries",
            "advance:SSPRK2 composition",
        ),
        ("Constant velocity on a periodic two-dimensional domain.",),
        (
            "Other velocity laws/boundaries and convergence under arbitrary refinements are not certified by the finite profile.",
        ),
        ("test_api_v02_models_M01_M14.py",),
    ),
    _p(
        "M02",
        "godunov-burgers",
        "models_M01_M14:build_m02",
        (
            "burgers-flux:Nonlinear conservative flux",
            "entropy-face:Godunov shock/rarefaction selection",
            "advance:SSPRK2 with shared faces",
        ),
        ("Periodic data with a transverse-invariant shock and rarefaction.",),
        (
            "General multidimensional wave interactions and entropy-solution convergence are separate qualifications.",
        ),
        ("test_api_v02_models_M01_M14.py",),
    ),
    _p(
        "M03",
        "stiffened-euler",
        "models_M01_M14:build_m03",
        (
            "balances:Mass, all momentum components and total energy",
            "eos:Authored stiffened-gas pressure and wave speeds",
            "advance:Conservative FV/SSPRK2",
        ),
        ("Two-dimensional periodic entropy wave; declared gamma and pressure offset.",),
        (
            "Strong shocks, near vacuum, arbitrary EOS and global positivity are not automatic consequences of API coverage.",
        ),
        ("test_api_v02_models_M01_M14.py",),
    ),
    _p(
        "M04",
        "oblique-tensor-diffusion",
        "models_M01_M14:build_m04",
        (
            "transport:Conservative transport",
            "tensor-diffusion:Full SPD tensor including mixed derivatives",
            "composition:Both terms at every SSPRK2 stage",
        ),
        ("Constant positive-definite tensor and periodic boundaries.",),
        (
            "Degenerate/variable tensors, general positivity and convergence require their own analysis and receipts.",
        ),
        ("test_api_v02_models_M01_M14.py",),
    ),
    _p(
        "M05",
        "two-dimensional-navier-stokes",
        "models_M01_M14:build_m05",
        (
            "euler:Compressible mass/momentum/energy fluxes",
            "stress:Viscous tensor and its mechanical work",
            "heat:Thermal conduction",
            "spatial:Both spatial directions and shared stencil faces",
        ),
        ("Ideal gas, constant viscosity/Prandtl, periodic two-dimensional perturbation.",),
        (
            "Wall/open boundaries, global positivity and continuum convergence remain separately qualified.",
        ),
        ("test_api_v02_models_M01_M14.py",),
        parameters={"two_dimensional": True},
        compatibility=(
            _named(
                "legacy-planar-navier-stokes",
                "models_M01_M14:build_m05",
                "Earlier initial profile invariant in the transverse direction.",
                two_dimensional=False,
            ),
        ),
    ),
    _p(
        "M06",
        "one-phase-stefan",
        "models_M01_M14:build_m06",
        (
            "enthalpy:Enthalpy-temperature map with latent plateau",
            "conduction:Heat flux",
            "boundary:Prescribed heat trace and boundary inventory",
            "advance:Conservative enthalpy evolution",
        ),
        (
            "One-phase planar Stefan problem, declared latent heat/capacity/conductivity, Dirichlet heating.",
        ),
        (
            "Curved/multiple interfaces and front convergence are physical/numerical extensions, not absent algebra primitives.",
        ),
        ("test_api_v02_models_M01_M14.py",),
    ),
    _p(
        "M07",
        "wet-dry-hydrostatic",
        "models_M01_M14:build_m07",
        (
            "water:Saint-Venant mass/momentum",
            "bed:Frozen topography and balanced bed-source deliveries",
            "dry-state:Explicit dry-state relation",
            "face:Hydrostatic reconstructed flux",
        ),
        ("Planar periodic lake at rest and short nonstationary transport, including dry cells.",),
        (
            "Long-time moving wetting fronts, dam breaks and general positivity demand independent regimes and evidence.",
        ),
        ("test_api_v02_models_M01_M14.py", "test_api_v02_drift_wetdry_native.py"),
    ),
    _p(
        "M08",
        "periodic-guiding-center",
        "models_M01_M14:build_m08",
        (
            "charge-transport:Advection by rotated potential gradient",
            "poisson:Global Poisson equation and mean gauge",
            "freshness:Fresh field at each stage and original residual",
        ),
        ("Signed zero-mean charge fluctuation on a periodic two-dimensional domain.",),
        (
            "Positivity of a distinct total density, other gauges/boundaries and long-time dynamics are not received by this profile.",
        ),
        ("test_api_v02_models_M01_M14.py",),
    ),
    _p(
        "M09",
        "magnetic-midpoint-schur",
        "models_magnetic_midpoint:build_m09_midpoint",
        (
            "euler:Full spatial Euler balances",
            "midpoint:Coupled electric/magnetic midpoint source",
            "current:Current enters the evolving potential equation",
            "schur:Exact authored elimination or monolithic relation",
            "original-residual:Original coupled residual and source energy checks",
        ),
        (
            "Periodic two-dimensional domain, declared stiff magnetic parameter, replicated finite spatial arrays.",
        ),
        (
            "Whole-splitting exact energy, arbitrary quasi-neutral limits and distributed scaling need separate receipts.",
        ),
        ("test_api_v02_magnetic_midpoint_native.py",),
        component_choices={"euler": {"source_only": False}},
        compatibility=(
            _named(
                "explicit-SSPRK2",
                "models_M01_M14:build_m09",
                "Earlier explicit magnetic source; does not implement the midpoint/Schur method.",
            ),
        ),
        companions=(
            _named(
                "monolithic-midpoint",
                "models_magnetic_midpoint:build_m09_midpoint",
                "Same source relation, solved monolithically.",
                solver="monolithic",
            ),
        ),
    ),
    _p(
        "M10",
        "scharfetter-gummel-poisson",
        "models_M01_M14:build_m10",
        (
            "drift-diffusion:Authored Scharfetter-Gummel face law",
            "poisson:Live potential constraint",
            "stage-coupling:Flux uses each fresh stage potential",
        ),
        ("Fixed coefficients and periodic boundaries; explicit selected timestep.",),
        (
            "Nonlinear mobilities, stiff entropy/positivity and alternative boundaries remain unestablished beyond received cases.",
        ),
        ("test_api_v02_models_M01_M14.py", "test_api_v02_drift_wetdry_native.py"),
    ),
    _p(
        "M11",
        "ternary-maxwell-stefan-poisson",
        "models_M01_M14:build_m11",
        (
            "mixture:Distinct species masses and charges",
            "friction:Coupled Maxwell-Stefan friction elimination",
            "mass-constraint:Exact common face mass constraint",
            "poisson:Fresh typed potential",
        ),
        ("Three strictly positive species, periodic finite-volume realization.",),
        (
            "Species disappearance and FE entropy theorems are not asserted for the selected centred FV discretization.",
        ),
        ("test_api_v02_models_M01_M14.py",),
    ),
    _p(
        "M12",
        "five-ten-moment-maxwell",
        "models_M12:build_m12",
        (
            "five-moments:Electron five-moment balances",
            "ten-moments:Ion pressure-tensor balances",
            "maxwell:Faraday/Ampere fields",
            "exchange:Shared Lorentz force and work",
            "constraints:Gauss/divergence diagnostics",
        ),
        ("Two spatial and three velocity components; vanishing central heat-flux closure.",),
        (
            "Nonlocal heat closure, general realizability and exact fully discrete RK4 energy are independent model/method questions.",
        ),
        ("test_api_v02_models_M01_M14.py",),
    ),
    _p(
        "M13",
        "streamer-photoionization",
        "models_M13:build_m13",
        (
            "transport:Charged-species drift/diffusion",
            "reactions:Explicit reaction stoichiometry",
            "poisson:Charge/potential coupling",
            "photoionization:Helmholtz radiation group",
            "invariants:Charge and nuclei inventories",
        ),
        ("Planar local-field coefficients and one photoionizing group.",),
        (
            "Dynamic electron energy, calibrated multigroup coefficients and physical front speed require separate models/evidence.",
        ),
        ("test_api_v02_models_M01_M14.py",),
    ),
    _p(
        "M14",
        "cold-sheath-circuit",
        "models_M14:build_m14",
        (
            "plasma:Cold plasma transport and potential",
            "wall:Absorbing physical face law",
            "circuit:Capacitor evolution",
            "charge-ledger:Same face current delivered to plasma and circuit",
        ),
        ("Prescribed reservoir and strictly outgoing absorbing wall.",),
        (
            "Thermal, reflecting and emitting sheaths are different closures; their energy and convergence are not claimed.",
        ),
        ("test_api_v02_models_M01_M14.py",),
    ),
    _p(
        "M15",
        "hyqmom-order-four",
        "models_M15_orders:build_m15_orders",
        (
            "moments:All declared raw-moment equations",
            "closure:Algorithm2 HyQMOM closure",
            "admissibility:Hankel-domain guards",
            "transport:Conservative moment flux",
        ),
        (
            "Selected order 4 with smooth positive finite-atom data; order 2 and Riemann variants remain available.",
        ),
        (
            "Vacuum/boundary-of-cone cases and universal convergence/admissibility require separate qualification.",
        ),
        ("test_api_v02_hyqmom_orders_native.py",),
        parameters={"order": 4},
    ),
    _p(
        "M16",
        "hyqmom15-b1",
        "models_M15_M18:build_m16",
        (
            "moments:Fifteen moment balances and B1 closure",
            "magnetic:Magnetic moment source",
            "transport:Declared directional fluxes",
            "admissibility:Declared domain and spectral refusal witness",
        ),
        (
            "B1 fifteen-moment model on its selected Cartesian regime; oblique universal hyperbolicity is false for this closure.",
        ),
        ("No general oblique evolution theorem is asserted.",),
        ("test_api_v02_models_M15_M21.py",),
        model_properties=(
            "The B1 closure has an oblique nonhyperbolic counterexample; this is a property of the physical closure, never a missing API component.",
        ),
    ),
    _p(
        "M17",
        "fan-li15-path",
        "models_M15_M18:build_m17",
        (
            "moments:Fan-Li15 moment and closure equations",
            "nonconservative:Authored path-dependent product",
            "path-quadrature:Straight raw-moment path with Gauss4 integration",
            "domain:Local spectral/SPD checks",
        ),
        ("Selected path and planar spatial variation.",),
        (
            "Other paths, multidimensional shock entropy, global realizability and SPD-boundary behavior are unestablished properties.",
        ),
        ("test_api_v02_models_M15_M21.py",),
    ),
    _p(
        "M18",
        "discrete-entropy-m2",
        "models_M18_entropy:build_m18_entropy",
        (
            "moments:Declared moment transport",
            "entropy:Exponential entropy dual and fixed velocity quadrature",
            "newton:Authored closure solve with original residual",
            "tangent:Closure derivatives for transport",
        ),
        (
            "Order 2 with Gauss6 on [-1,1], interior realizable moments; order 3 is a named parameter variant.",
        ),
        (
            "Continuous-velocity closure, near-boundary Newton convergence and general realizability/entropy stability are not certified.",
        ),
        ("test_api_v02_entropy_MN_native.py",),
        parameters={"order": 2, "quadrature": 6},
    ),
)

_MODELS += (
    _p(
        "M19",
        "distributed-vlasov-poisson-bgk",
        "models_kinetic_nd:build_kinetic_nd",
        (
            "phase-space:Actual distributed x*v product with independent incidences",
            "vlasov:Transport in every x/v direction",
            "moments:Weighted velocity-fibre reductions and field lift",
            "poisson:Global configuration-space solve",
            "bgk:Moment-corrected BGK collision",
            "acceptance:Original residual/charge/positivity checks",
        ),
        (
            "Periodic 3x3v, finite zero-flux velocity box, one species plus neutralizing background, explicit FV/Euler.",
        ),
        (
            "A finite-box moment correction does not imply an H theorem, exact whole-step energy or stiff BGK AP behavior.",
            "Velocity truncation/convergence and scale/GPU profiles need their own receipts.",
        ),
        ("test_api_v02_kinetic_nd.py",),
        parameters={
            "model": "M19",
            "configuration_shape": (3, 3, 3),
            "velocity_shape": (4, 4, 4),
            "masses": (1.0,),
            "charges": (-1.0,),
        },
        compatibility=(
            _named(
                "legacy-1x1v",
                "models_M15_M28:build_m19",
                "Original finite 1x1v Poisson/BGK profile.",
            ),
        ),
    ),
    _p(
        "M20",
        "distributed-mixture-fokker-planck",
        "models_kinetic_nd:build_kinetic_nd",
        (
            "phase-space:Distributed multidimensional distributions",
            "spatial-transport:Conservative transport in every configuration-space direction",
            "pairs:Every self/cross collision pair with species parameters",
            "drift-diffusion:Velocity-space Fokker-Planck flux",
            "invariants:Per-species number and joint momentum/energy correction",
            "pair-fold:Replicated finite pair fold with bounded temporary lifetimes",
            "acceptance:No negative candidate is silently clipped",
        ),
        (
            "Three-species 3x3v symmetric mixture subclass of Habbershaw; finite zero-flux velocity box and explicit FV/Euler.",
        ),
        (
            "General asymmetric coefficient fitting and the paper's implicit solver are distinct variants.",
            "Projected finite quadrature is not unconditionally positive or entropic; counterexamples remain documented.",
        ),
        ("test_api_v02_kinetic_nd.py",),
        parameters={
            "model": "M20",
            "configuration_shape": (3, 3, 3),
            "velocity_shape": (4, 4, 4),
            "masses": (1.0, 1.5, 2.0),
            "charges": (-1.0, 1.0, 1.0),
        },
        compatibility=(
            _named(
                "legacy-two-species-1x1v",
                "models_M20_M21:build_m20",
                "Original inhomogeneous two-species 1x1v profile.",
            ),
        ),
    ),
    _p(
        "M21",
        "full-f-geometric-nonboussinesq",
        "model_gyrokinetic_nd:build_gyrokinetic_nd",
        (
            "phase-space:Distributed 3x2v full g=Jf",
            "characteristics:Parallel/ExB/mirror/curvature/flutter transport",
            "polarization:Density-dependent perpendicular operator",
            "hamiltonian:Second-order ExB energy accompanies nonlinear polarization",
            "ampere:Global A_parallel constraint",
            "induction:Same-discrete-velocity dA_parallel/dt coupling",
            "collisions:Parallel/magnetic-moment collision flux and invariant projection",
        ),
        (
            "Stationary periodic planar magnetic geometry with b·curl(b)=0, J=B and actual three-dimensional spatial operators.",
            "Declared FV/Euler, finite zero-flux velocity box and explicit collision moment projection.",
        ),
        (
            "Arbitrary tokamak metrics and Mandell's DG energy/Alfven proofs are not claimed for this FV realization.",
            "Moment projection does not provide unconditional positivity; Assess and negative witnesses retain this distinction.",
        ),
        ("test_api_v02_gyrokinetic_nd.py",),
        parameters={"nonlinear_polarization": True, "collision_frequency": 0.02},
        component_choices={
            "polarization": {"nonlinear_polarization": True},
            "hamiltonian": {"nonlinear_polarization": True},
        },
        compatibility=(
            _named(
                "legacy-slab-fixed-kperp",
                "models_M20_M21:build_m21",
                "Original slab with fixed perpendicular symbol and Boussinesq coefficient.",
            ),
        ),
        companions=(
            _named(
                "linear-polarization",
                "model_gyrokinetic_nd:build_gyrokinetic_nd",
                "Consistent linear coefficient with H2 omitted.",
                nonlinear_polarization=False,
                collision_frequency=0.02,
            ),
        ),
    ),
    _p(
        "M22",
        "gray-m1-material",
        "models_M22_M25_M28:build_m22",
        (
            "material:Euler material balances",
            "radiation:M1 energy/flux and closure",
            "exchange:Shared opposite energy/momentum exchange",
            "stages:Spatial fluxes and exchange at each stage",
        ),
        (
            "Gray lab-frame closure with declared c/a/cv, periodic two-dimensional mesh and explicit opacity/timestep.",
        ),
        (
            "Comoving/relativistic corrections, diffusion-limit AP, stiff opacity and Marshak layers require separately stated variants and receipts.",
        ),
        ("test_api_v02_continuum_native.py",),
    ),
    _p(
        "M23",
        "hall-ohm-staggered-ct",
        "model_hall_mhd_ct:build_hall_mhd_ct",
        (
            "material:Mass, all momentum components and total energy",
            "stress:Conservative Maxwell stress",
            "ct:Face magnetic fields and div(curl)=0 incidence",
            "hall:Skew Hall electric work through adjoint interpolation",
            "ohm:Distinct resistive electric field and heating budget",
            "poynting:Full energy transport and curl-adjoint identity",
            "advance:Declared RK4 composition",
        ),
        (
            "Nonlinear periodic Cartesian 2.5D adiabatic Hall/Ohm variant, explicit selected coefficients/timestep.",
            "Hall energy transport follows Kimura-Morrison; no Hall heating or artificial Ohmic increase is hidden.",
        ),
        (
            "Arbitrary magnetic boundaries, moving unstructured mesh, Hall-stiff stability and shock admissibility are not asserted.",
        ),
        ("test_api_v02_hall_mhd_ct.py",),
        compatibility=(
            _named(
                "legacy-planar-whistler",
                "models_M22_M25_M28:build_m23",
                "Earlier complete flux equations but only planar divB topology and total damping reception.",
            ),
        ),
    ),
    _p(
        "M24",
        "bn-pressure-velocity-relaxation",
        "models_M22_M25_M28:build_m24",
        (
            "balances:Seven-equation two-phase balances",
            "interface:Declared interfacial pressure/velocity",
            "path:Nonconservative path and quadrature",
            "pressure-relaxation:Paired pressure relaxation with interfacial work",
            "velocity-relaxation:Opposite momentum exchange and paired energy work",
            "stages:Same stages for conservative/nonconservative/source contributions",
        ),
        (
            "p_I=p1, u_I=u2, gamma=(1.4,1.6), straight conserved-state Gauss2 path and explicit positive relaxation rates.",
            "Relaxing variant is explicitly additional to the convective S18 formulation.",
        ),
        (
            "Phase disappearance, stiff asymptotic limits, universal contact preservation and positivity require separate qualification.",
        ),
        ("test_api_v02_continuum_native.py",),
        parameters={"pressure_relaxation": 0.2, "velocity_relaxation": 0.3},
        compatibility=(
            _named(
                "legacy-convective",
                "models_M22_M25_M28:build_m24",
                "Seven-equation convective variant with both relaxation coefficients zero.",
                pressure_relaxation=0.0,
                velocity_relaxation=0.0,
            ),
        ),
    ),
    _p(
        "M25",
        "gpr-thermomechanical",
        "models_M22_M25_M28:build_m25",
        (
            "mechanics:Mass, three momentum components and total energy",
            "distortion:Evolved distortion tensor and nonconservative transport",
            "thermal:Evolved thermal impulse and heat flux",
            "relaxations:Distortion and thermal relaxation laws",
            "path:Declared path and face quadrature",
        ),
        (
            "Seventeen components, two spatial axes, small planar perturbation, explicit relaxation and declared Rusanov speed 64.",
        ),
        (
            "That wave-speed bound is not universal; stiff viscous/thermal limits and exact det(A)=rho/rho0 need separate proofs or receipts.",
        ),
        ("test_api_v02_continuum_native.py",),
    ),
    _p(
        "M26",
        "aggregation-global-convolution",
        "models_M15_M28:build_m26",
        (
            "potential:Authored aggregation interaction law",
            "convolution:Global periodic convolution",
            "diffusion:Density-dependent drift/diffusion flux",
            "balance:Conservative spatial update and energy observable",
        ),
        (
            "Smooth cosine kernel, actual multicell PDE, selected planar periodic perturbation and explicit timestep.",
        ),
        (
            "Singular kernels, arbitrary-dimensional reception, unconditional energy/positivity and continuum convergence are separate obligations.",
        ),
        ("test_api_v02_spatial_pdes.py",),
    ),
    _p(
        "M27",
        "cahn-hilliard-fourth-order",
        "models_M15_M28:build_m27",
        (
            "free-energy:Declared double-well energy and gradient penalty",
            "chemical-potential:Authored variational chemical potential",
            "fourth-order:True spatial fourth-order composition",
            "balance:Conservative mobility flux and energy observable",
        ),
        ("Constant mobility, periodic planar profile and explicit selected timestep.",),
        (
            "Degenerate mobility, stiff interface limits and unconditional energy stability are not properties of the language.",
        ),
        ("test_api_v02_spatial_pdes.py",),
    ),
    _p(
        "M28",
        "multidimensional-reynolds-ale",
        "model_ale_nd:build_ale_nd",
        (
            "geometry:Time-dependent multidimensional cell/face measures",
            "swept-measures:Endpoint-consistent swept face volumes",
            "reynolds:Relative physical/mesh transport flux",
            "gcl:Same interval enforces geometric conservation",
            "inventory:Volume-weighted conserved inventory",
        ),
        (
            "Periodic Cartesian deformations with positive Jacobian; both spatial velocities and nonuniform transported data are active.",
        ),
        (
            "Arbitrary topology changes, moving-face quadratures and geometry singularities need separately declared algorithms and evidence.",
        ),
        ("test_api_v02_ale_nd.py",),
        compatibility=(
            _named(
                "legacy-one-dimensional-ALE",
                "models_M22_M25_M28:build_m28",
                "Original Dim1 endpoint-Reynolds moving-volume profile.",
            ),
        ),
    ),
)

_METHODS = (
    _p(
        "T01",
        "stage-constrained-ssprk2",
        "methods_T01_T12:build_t01",
        (
            "tableau:Authored RK stages and weights",
            "constraints:Fresh Poisson solve at each stage",
            "stage-state:Correct stage state/clock passed to each operator",
            "original-residual:Field residual retained",
        ),
        ("SSPRK2 on a periodic guiding-center PDE.",),
        (
            "General RK tableaux, arbitrary constrained order and nonperiodic stage boundaries require their own compositions/receipts.",
        ),
        ("test_api_v02_methods_T01_T12.py",),
    ),
    _p(
        "T02",
        "ars222-imex",
        "methods_coupled_nonsymmetric:build_t02_imex_second_order",
        (
            "split:Distinct explicit and implicit rates",
            "tableau:Declared second-order additive coefficients",
            "implicit:Global implicit spatial solves",
            "combination:Both rates contribute with correct stage times/weights",
        ),
        ("Selected ARS222 finite periodic spatial system with declared stiff linear operator.",),
        (
            "All higher-order conditions, nonlinear stiff order and AP limits are not inferred from this method witness.",
        ),
        ("test_api_v02_coupled_methods_native.py",),
        compatibility=(
            _named(
                "legacy-IMEX-Euler",
                "methods_T01_T12:build_t02",
                "First-order additive explicit-decay/implicit-diffusion profile.",
            ),
        ),
    ),
    _p(
        "T03",
        "dirichlet-extended-strang",
        "method_splitting_boundary:build_t03_dirichlet",
        (
            "lifting:Time-dependent physical boundary lifting",
            "domains:Each subproblem preserves its declared trace domain",
            "flows:Noncommuting diffusion/reaction subflows",
            "clocks:Extended physical clock distinct from algorithmic durations",
            "composition:Strang half/full/half flow sequence",
        ),
        (
            "Manufactured spatial Dirichlet heat/reaction PDE; CN diffusion subflow and exact local reaction.",
        ),
        (
            "General boundary-domain commutator estimates and nonlinear splitting stability are not universally certified.",
        ),
        ("test_api_v02_splitting_boundary_native.py",),
        compatibility=(
            _named(
                "legacy-nonautonomous-ODE",
                "methods_T01_T12:build_t03",
                "Earlier extended-clock ODE splitting witness.",
            ),
        ),
    ),
    _p(
        "T04",
        "global-nonsymmetric-schur",
        "methods_coupled_nonsymmetric:build_t04_nonsymmetric",
        (
            "blocks:Coupled global block relation",
            "elimination:Nonlocal block inverse and Schur action",
            "solver:Authored nonsymmetric Krylov algorithm",
            "reconstruction:Reconstruct eliminated variables",
            "residual:Original coupled residual, not only Schur residual",
        ),
        (
            "Periodic declared nonsymmetric invertible finite operators; solve tolerance remains explicit.",
        ),
        (
            "Arbitrary singular/indefinite spectra, preconditioners and nonperiodic inherited Schur boundaries require their own algorithms and receipts.",
        ),
        ("test_api_v02_coupled_methods_native.py",),
        compatibility=(
            _named(
                "legacy-CG-Schur",
                "methods_T01_T12:build_t04_global",
                "Earlier symmetric periodic two-component CG/Schur system.",
            ),
        ),
    ),
    _p(
        "T05",
        "nested-entropy-newton",
        "methods_T01_T12:build_t05",
        (
            "closure:Inner entropy relation",
            "outer:Outer nonlinear residual uses inner solve",
            "tangent:Implicit tangent of the selected inner solution",
            "termination:Original residual and bounded iteration/failure",
        ),
        ("Strictly realizable fixed two-velocity quadrature and selected convex root.",),
        (
            "Global feasibility, singular Jacobians and differentiation across root switching need explicit scientific assumptions.",
        ),
        ("test_api_v02_methods_T01_T12.py",),
    ),
    _p(
        "T06",
        "global-selected-relation",
        "methods_coupled_nonsymmetric:build_t06_selected_global",
        (
            "arguments:Old captures and unknowns remain distinct",
            "global-relation:Spatially coupled nonlinear relation",
            "branch:Explicit selected root/domain",
            "residual:Original relation checked after solve",
        ),
        (
            "Finite declared positive/negative branches with explicit old-state versus unknown argument mode.",
        ),
        (
            "Automatic physical root selection and global convergence across arbitrary multiple/singular roots are not promised.",
        ),
        ("test_api_v02_coupled_methods_native.py",),
        compatibility=(
            _named(
                "legacy-local-offdiagonal",
                "methods_T01_T12:build_t06",
                "Local F(a,b)=a*b versus b² distinction.",
            ),
        ),
    ),
    _p(
        "T07",
        "variable-bdf-authenticated-history",
        "methods_T01_T12:build_t07",
        (
            "startup:Explicit startup method",
            "coefficients:Variable-step BDF coefficients",
            "history:Actual state/step/increment history",
            "intervals:Authenticated HistoryInterval origins",
            "defect:Weighted historical production and original finite-solve defect",
        ),
        (
            "BE startup followed by selected nonuniform BDF2 steps; intervals name retained publications.",
        ),
        (
            "Stability for arbitrary step-ratio sequences and higher orders is separate from implementation coverage.",
        ),
        ("test_api_v02_methods_T01_T12.py", "test_api_v02_history_transfer_native.py",
         "test_api_v02_spatial_bdf2_native.py"),
        companions=(
            _named(
                "global-spatial-BDF2",
                "method_spatial_bdf2:build_spatial_bdf2",
                "Authored global implicit spatial recurrence, partial AMR, order and restart receivers.",
            ),
            _named(
                "BDF3-after-AMR-history-transfer",
                "method_history_transfer:build_history_transfer",
                "Quadratic authored transfer and a nonuniform BDF3 consumer; separate native receipt.",
            ),
        ),
    ),
    _p(
        "T08",
        "recursive-amr-window",
        "methods_T01_T12:build_t08",
        (
            "recursion:Parent/child time windows and intermediate states",
            "subcycling:Declared coarse/fine clock relation",
            "face-history:Temporal face inventories",
            "synchronization:Reflux and average-down",
            "rollback:Parent-window rejection restores pre-attempt state",
        ),
        ("Two-level periodic Cartesian advection and declared 2:1 subcycling.",),
        (
            "Other physical boundaries, nonlinear inter-child couplings and arbitrary hierarchies need separate receipts.",
        ),
        ("test_api_v02_methods_T01_T12.py", "test_api_v02_history_transfer_native.py"),
        companions=(
            _named(
                "state-history-regrid",
                "method_history_transfer:build_history_transfer",
                "Actual new coverage and quadratic temporal history transfer, with rejection witnesses.",
            ),
            _named(
                "flux-history-regrid",
                "method_history_transfer:build_flux_history_transfer",
                "Time-quadratic spatial flux, RK3 startup/AB3 and three transferred RHS slots.",
            ),
        ),
    ),
    _p(
        "T09",
        "spatial-duhamel-interval",
        "method_interval_transport:build_interval_transport",
        (
            "characteristics:Spatial characteristic reconstruction",
            "duhamel:Duration-dependent local integral solution",
            "interval:Actual requested interval passed as SSA value",
            "joint-uses:Owned face and collision contributions share interval law",
            "retry:Retry reevaluates interval-dependent equations",
        ),
        ("Two velocities, periodic transport and BGK equilibrium frozen on each interval.",),
        (
            "Self-consistent nonlinear Maxwellians and fluid/diffusive AP limits need additional selected equations and qualification.",
        ),
        ("test_api_v02_interval_transport_native.py", "test_api_v02_nonlinear_interval_transport_native.py"),
        companions=(
            _named(
                "nonlinear-interval-equilibrium",
                "method_nonlinear_interval_transport:build_nonlinear_interval_transport",
                "Nonlinear reconstructed equilibrium, native duration request and rejected-step recomputation.",
            ),
        ),
        compatibility=(
            _named(
                "legacy-homogeneous-Duhamel",
                "methods_T01_T12:build_t09_homogeneous",
                "Earlier homogeneous finite-edge Duhamel witness.",
            ),
        ),
    ),
    _p(
        "T10",
        "whole-macro-relaxation",
        "method_relaxed_composite:build_t10_composite",
        (
            "macro-entry:Actual snapshot at entry of whole macro step",
            "observable:Global volume-weighted relaxation observable",
            "factor:Authored relaxation factor",
            "state-time:Same factor changes state and reached clock",
            "exchange-ledger:Same factor finalizes face exchange and reflux",
            "rollback:Rejected attempt restores whole window",
        ),
        (
            "Rotation plus spatial passive transport on AMR; chosen quadratic observable measures rotation components only.",
        ),
        (
            "This observable does not prove arbitrary transport quadratic-energy conservation; multiple-invariant branch selection remains a separate method.",
        ),
        (
            "test_api_v02_relaxed_composite_native.py",
            "test_api_v02_composite_integral_native.py",
            "test_api_v02_relaxed_multiblock_native.py",
            "test_api_v02_relaxed_history_native.py",
            "test_api_v02_relaxed_cutcell_native.py",
            "test_api_v02_authored_stability_native.py",
        ),
        parameters={"temporal_ratio": 2},
        component_choices={"factor": {"relaxed": True}},
        companions=(
            _named(
                "cutcell-volume-relaxation",
                "method_relaxed_cutcell:build_t10_cutcell",
                "Authored observable consumes actual cut-cell measures; late rejection and restart are explicit.",
            ),
            _named(
                "authored-energy-admission",
                "method_authored_stability:build_authored_stability",
                "Python-defined energy inequality compiles into numerical admission and bounded retry.",
            ),
        ),
        compatibility=(
            _named(
                "legacy-homogeneous-Heun",
                "methods_T01_T12:build_t10",
                "Earlier homogeneous Heun rotation with gamma shared by state/time.",
            ),
        ),
    ),
    _p(
        "T11",
        "network-mprk22",
        "methods_patankar_network:build_t11_network",
        (
            "processes:Explicit directed production/destruction graph",
            "weights:Authored modified Patankar weights",
            "stages:Declared second-order MPRK22 stages",
            "shared-transfers:One transfer serves donor and receiver",
            "zero-inventories:Declared behavior at empty inventories",
        ),
        (
            "Finite nonnegative network with explicit process presentation; empty-inventory and cancelling-circulation variants.",
        ),
        (
            "Equivalence of every process presentation, arbitrary higher order and universal positivity beyond declared assumptions are not asserted.",
        ),
        ("test_api_v02_coupled_methods_native.py",),
        compatibility=(
            _named(
                "legacy-two-species-Patankar-Euler",
                "methods_T01_T12:build_t11",
                "Earlier two-species first-order directed-transfer method.",
            ),
        ),
    ),
    _p(
        "T12",
        "versioned-amr-face-replacement",
        "method_flux_correction_amr:build_t12_amr",
        (
            "candidate:High/low shared face candidates",
            "selection:Authored admissibility selection",
            "replacement:Replace one versioned face contribution, never add twice",
            "incidence:Rebuild all incident deliveries",
            "amr-ledger:Accepted replacement enters actual coarse/fine reflux ledger",
        ),
        (
            "Selected one-pass positivity correction on periodic Cartesian AMR; actual coarse/fine interface.",
        ),
        (
            "Arbitrary EOS, iterative MOOD and universal admissibility are not certified by these witnesses.",
            "The dependent-field profile and Euler correction are distinct compositions requiring their own receipts.",
        ),
        (
            "test_api_v02_stencil_amr_native.py",
            "test_api_v02_stencil_selection_native.py",
            "test_api_v02_corrected_field_native.py",
            "test_api_v02_euler_correction_native.py",
        ),
        component_choices={"replacement": {"selected": True}},
        compatibility=(
            _named(
                "legacy-finite-edge",
                "methods_T01_T12:build_t12_finite_edge",
                "Original co-located finite-edge replacement; no spatial receipt transfer.",
            ),
            _named(
                "uniform-spatial-face",
                "method_flux_correction:build_t12_meshface",
                "Uniform Cartesian scalar face replacement.",
            ),
        ),
        companions=(
            _named(
                "field-after-replacement",
                "method_flux_correction:build_t12_corrected_field",
                "Recompute the dependent field from the accepted corrected state, with stale-source negative witness.",
            ),
            _named(
                "compressible-euler-replacement",
                "method_euler_correction:build_euler_correction",
                "Shared-face correction using both incident provisional Euler states; distinct from the scalar AMR profile.",
            ),
        ),
    ),
)

SELECTED_PROFILES = MappingProxyType(
    {profile.identifier: profile for profile in (*_MODELS, *_METHODS)}
)
if set(SELECTED_PROFILES) != {
    *(f"M{i:02}" for i in range(1, 29)),
    *(f"T{i:02}" for i in range(1, 13)),
}:
    raise RuntimeError(
        "the selected-profile contract catalogue must contain exactly 28 models and 12 methods"
    )


def selected_profile(identifier):
    """Select a declared profile; this does not assert its implementation or execution."""
    try:
        return SELECTED_PROFILES[str(identifier).upper()]
    except KeyError as error:
        raise ValueError("unknown corpus identifier: " + str(identifier)) from error


def named_profiles(identifier):
    contract = selected_profile(identifier)
    return (
        NamedProfile(contract.variant, contract.builder, "Selected equation/composition contract."),
        *contract.compatibility_profiles,
        *contract.companion_profiles,
    )
