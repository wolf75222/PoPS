"""Typed native route IDs (ADC-584): the single Python registry of behavior routes.

Every algorithmic choice (Riemann flux, limiter, reconstructed variables, time treatment,
splitting, field solver, Poisson boundary condition, layout, model bricks) is identified by a
typed :class:`Route`, not by a free string. The typed descriptors (``pops.numerics`` /
``pops.solvers`` / the time bricks) lower to these Routes; the ONLY places that emit the legacy
wire token toward the C++ ABI are the bounded adapters (``pops.runtime._system_install`` and
``pops.codegen.compile_emit``), and they emit ``str(route)`` -- a :class:`Route` IS its wire
token (``str`` subclass), so the crossing stays byte-identical while the identity, requirements,
limitations and native entry point become typed and inspectable.

This module is the MIRROR of ``include/pops/runtime/config/route_ids.hpp`` (same families, same
tokens, same order); ``tests/architecture/test_route_registry_parity.py`` locks the two at the
source level, and the C++ static_asserts lock route_ids.hpp against the historical tag tables
(kLimiters / kRiemanns / kTransports / kSources / kElliptics). Deliberately IMPORT-FREE (stdlib
only): the architecture gate loads it standalone, without the compiled ``_pops`` module.
"""


class Route(str):
    """A typed native route ID.

    ``str`` subclass: ``str(route)`` (and any legacy comparison against the wire token) stays
    byte-identical to the historical string, so the pybind/ABI crossings and the existing tests
    keep working unchanged. The typed surface on top:

    - ``route.family`` / ``route.id``: the typed identity (``"riemann"`` / ``"riemann.hll"``);
    - ``route.native_entry``: the native C++ entry point (``"pops::HLLFlux"``);
    - ``route.requirements`` / ``route.limitations``: the declared route contract (documentary;
      the hard guards stay at the install/bind sites, which now cite these fields).

    Instances are created ONLY by this module's registry: an unknown token never constructs a
    Route (:func:`resolve` refuses it), so a Route value is a proof of validation.
    """

    # Attribute declarations (set in __new__; str subclasses cannot use nonempty __slots__).
    family: str
    id: str
    native_entry: str
    requirements: tuple
    limitations: tuple

    def __new__(cls, family, token, native_entry, requirements=(), limitations=()):
        self = super().__new__(cls, token)
        self.family = family
        self.id = "%s.%s" % (family, token)
        self.native_entry = native_entry
        self.requirements = tuple(requirements)
        self.limitations = tuple(limitations)
        return self

    @property
    def token(self):
        """The legacy wire token (the ``str`` value itself; debug / ABI / messages only)."""
        return str(self)

    def manifest(self):
        """The structured manifest row of this route (inspection / reports)."""
        return {
            "family": self.family,
            "id": self.id,
            "token": str(self),
            "native_entry": self.native_entry,
            "requirements": list(self.requirements),
            "limitations": list(self.limitations),
        }

    def __repr__(self):
        return "Route(%s)" % self.id


def _split(csv):
    return tuple(s for s in csv.split(",") if s) if csv else ()


# --- The route tables: MIRROR of route_ids.hpp, one row per line, same order. ------------------
# (family, ((token, native_entry, requirements_csv, limitations_csv), ...))
_TABLES = {
    "riemann": (
        ("rusanov", "pops::RusanovFlux", "max_wave_speed", ""),
        ("hll", "pops::HLLFlux", "physical_flux,wave_speeds", ""),
        ("hllc", "pops::HLLCFlux",
         "physical_flux,pressure,wave_speeds,contact_speed,hllc_star_state",
         "polar geometry not wired; canonical path assumes 2D Euler unless HasHLLCStructure"),
        ("roe", "pops::RoeFlux", "physical_flux,roe_average",
         "polar geometry not wired; canonical path assumes 2D Euler unless HasRoeDissipation"),
    ),
    "limiter": (
        ("none", "pops::NoSlope", "", ""),
        ("minmod", "pops::Minmod", "", ""),
        ("vanleer", "pops::VanLeer", "", ""),
        ("weno5", "pops::Weno5Z", "3-cell halo",
         "prototype backend not wired (host order-1 residual)"),
    ),
    "recon": (
        ("conservative", "pops::make_block(recon_prim=false)", "", ""),
        ("primitive", "pops::make_block(recon_prim=true)", "primitive_vars",
         "requires a model exposing primitive variables"),
    ),
    "time": (
        ("explicit", "pops::SSPRK2", "", ""),
        ("ssprk3", "pops::SSPRK3", "",
         "aot .so ABI not wired (SSPRK2-only extern C entry); native add_block/add_native_block "
         "only"),
        ("euler", "pops::ForwardEuler", "",
         "aot .so ABI not wired; native add_block/add_native_block only; validation use, never "
         "default"),
        ("imex", "pops::AdvanceImex", "implicit source term", ""),
        ("imexrk_ars222", "pops::ImexRkArs222", "implicit source term",
         "composed native add_block only (.so ABIs do not carry the RK tableau)"),
    ),
    "splitting": (
        ("lie", "pops::SystemStepper(lie)", "", ""),
        ("strang", "pops::SystemStepper(strang)", "",
         "H(dt/2) S(dt) H(dt/2); requires a condensed source stage"),
    ),
    "field_solver": (
        ("geometric_mg", "pops::GeometricMG", "", ""),
        ("fft", "pops::PoissonFFTSolver", "periodic bc,constant coefficient",
         "walls / variable epsilon not wired; non power-of-two grid falls back to O(n^2) DFT"),
        ("fft_spectral", "pops::PoissonFFTSolver(spectral)",
         "periodic bc,constant coefficient",
         "walls / variable epsilon not wired; continuous symbol -(kx^2+ky^2)"),
        ("polar", "pops::PolarPoissonSolver", "polar geometry",
         "annular polar only (r_min > 0)"),
    ),
    "poisson_bc": (
        ("auto", "resolved from the wall/periodic system config", "", ""),
        ("periodic", "pops::fill_boundary(periodic)", "", ""),
        ("dirichlet", "pops::PhysicalBc(dirichlet)", "", ""),
        ("neumann", "pops::PhysicalBc(neumann)", "", ""),
    ),
    "layout": (
        ("uniform", "pops::System", "", ""),
        ("amr", "pops::AmrSystem", "",
         "refinement ratio 2 (kAmrRefRatio); fft field solver not wired"),
    ),
    "transport": (
        ("exb", "pops::ExBVelocity", "", "scalar (1 var); no fluid source"),
        ("compressible", "pops::CompressibleFlux", "", "polar geometry not wired"),
        ("isothermal", "pops::IsothermalFlux", "", ""),
    ),
    "source": (
        ("none", "pops::NoSource", "", ""),
        ("potential", "pops::PotentialForce", "fluid transport (>= 3 vars)", ""),
        ("gravity", "pops::GravityForce", "fluid transport (>= 3 vars)", ""),
        ("magnetic", "pops::MagneticLorentzForce",
         "fluid transport (>= 3 vars),aux B_z channel",
         "explicit regime (stiff regime -> condensed Schur stage)"),
        ("potential_magnetic",
         "pops::CompositeSource<PotentialForce, MagneticLorentzForce>",
         "fluid transport (>= 3 vars),aux B_z channel", ""),
    ),
    "elliptic": (
        ("charge", "pops::ChargeDensity", "", ""),
        ("background", "pops::BackgroundDensity", "", ""),
        ("gravity", "pops::GravityCoupling", "", ""),
    ),
    "source_stage": (
        ("electrostatic_lorentz", "pops::ElectrostaticLorentzCondensedSchur",
         "magnetic field B_z,system potential phi", "theta in (0, 1]"),
    ),
    "poisson_rhs": (
        ("charge_density", "per-block ChargeDensity bricks summed", "",
         "alias of composite when every block carries a charge density (bit-identical)"),
        ("composite", "per-block elliptic bricks summed", "", ""),
    ),
    "wall": (
        ("none", "no wall (fully periodic/physical domain)", "", ""),
        ("circle", "pops::make_wall_predicate(circle)", "wall_radius > 0", ""),
    ),
}

# Historical alias spellings, resolved by resolve() to their canonical route (parse-only
# compatibility, mirror of parse_source_route / parse_time_route in route_ids.hpp): the wire
# token emitted is always the canonical spelling.
_ALIASES = {
    "source": {"lorentz": "magnetic", "potential_lorentz": "potential_magnetic"},
    "time": {"ssprk2": "explicit"},
}

_REGISTRY = {
    family: {token: Route(family, token, entry, _split(req), _split(lim))
             for (token, entry, req, lim) in rows}
    for family, rows in _TABLES.items()
}


def resolve(family, token, context="routes"):
    """Resolve a wire @p token to its typed :class:`Route` -- refuse an unknown one, never default.

    The refusal cites the requested descriptor token, the family and the valid route set
    (ADC-584: an unknown or unsupported route is refused before bind, it never falls back).
    """
    routes = _REGISTRY.get(family)
    if routes is None:
        raise ValueError("%s: unknown route family %r (valid: %s)"
                         % (context, family, "|".join(sorted(_REGISTRY))))
    canonical = _ALIASES.get(family, {}).get(token, token)
    route = routes.get(canonical)
    if route is None:
        raise ValueError(
            "%s: unknown %s route %r (valid: %s); typed routes never fall back to a default"
            % (context, family, token, "|".join(routes)))
    return route


def routes_of(family):
    """The ordered typed routes of @p family (registry order = route_ids.hpp order)."""
    return tuple(_REGISTRY[family].values())


def route_manifest():
    """The full structured route manifest (every family, registry order) -- inspection surface."""
    return [route.manifest() for family in _TABLES for route in _REGISTRY[family].values()]


# --- Typed route constants (the internal currency of the lowering layer) -----------------------
RIEMANN_RUSANOV = _REGISTRY["riemann"]["rusanov"]
RIEMANN_HLL = _REGISTRY["riemann"]["hll"]
RIEMANN_HLLC = _REGISTRY["riemann"]["hllc"]
RIEMANN_ROE = _REGISTRY["riemann"]["roe"]

LIMITER_NONE = _REGISTRY["limiter"]["none"]
LIMITER_MINMOD = _REGISTRY["limiter"]["minmod"]
LIMITER_VANLEER = _REGISTRY["limiter"]["vanleer"]
LIMITER_WENO5 = _REGISTRY["limiter"]["weno5"]

RECON_CONSERVATIVE = _REGISTRY["recon"]["conservative"]
RECON_PRIMITIVE = _REGISTRY["recon"]["primitive"]

TIME_EXPLICIT = _REGISTRY["time"]["explicit"]
TIME_SSPRK3 = _REGISTRY["time"]["ssprk3"]
TIME_EULER = _REGISTRY["time"]["euler"]
TIME_IMEX = _REGISTRY["time"]["imex"]
TIME_IMEXRK_ARS222 = _REGISTRY["time"]["imexrk_ars222"]

SPLITTING_LIE = _REGISTRY["splitting"]["lie"]
SPLITTING_STRANG = _REGISTRY["splitting"]["strang"]

FIELD_SOLVER_GEOMETRIC_MG = _REGISTRY["field_solver"]["geometric_mg"]
FIELD_SOLVER_FFT = _REGISTRY["field_solver"]["fft"]
FIELD_SOLVER_FFT_SPECTRAL = _REGISTRY["field_solver"]["fft_spectral"]
FIELD_SOLVER_POLAR = _REGISTRY["field_solver"]["polar"]

POISSON_BC_AUTO = _REGISTRY["poisson_bc"]["auto"]
POISSON_BC_PERIODIC = _REGISTRY["poisson_bc"]["periodic"]
POISSON_BC_DIRICHLET = _REGISTRY["poisson_bc"]["dirichlet"]
POISSON_BC_NEUMANN = _REGISTRY["poisson_bc"]["neumann"]

LAYOUT_UNIFORM = _REGISTRY["layout"]["uniform"]
LAYOUT_AMR = _REGISTRY["layout"]["amr"]

TRANSPORT_EXB = _REGISTRY["transport"]["exb"]
TRANSPORT_COMPRESSIBLE = _REGISTRY["transport"]["compressible"]
TRANSPORT_ISOTHERMAL = _REGISTRY["transport"]["isothermal"]

SOURCE_NONE = _REGISTRY["source"]["none"]
SOURCE_POTENTIAL = _REGISTRY["source"]["potential"]
SOURCE_GRAVITY = _REGISTRY["source"]["gravity"]
SOURCE_MAGNETIC = _REGISTRY["source"]["magnetic"]
SOURCE_POTENTIAL_MAGNETIC = _REGISTRY["source"]["potential_magnetic"]

ELLIPTIC_CHARGE = _REGISTRY["elliptic"]["charge"]
ELLIPTIC_BACKGROUND = _REGISTRY["elliptic"]["background"]
ELLIPTIC_GRAVITY = _REGISTRY["elliptic"]["gravity"]

SOURCE_STAGE_ELECTROSTATIC_LORENTZ = _REGISTRY["source_stage"]["electrostatic_lorentz"]

POISSON_RHS_CHARGE_DENSITY = _REGISTRY["poisson_rhs"]["charge_density"]
POISSON_RHS_COMPOSITE = _REGISTRY["poisson_rhs"]["composite"]

WALL_NONE = _REGISTRY["wall"]["none"]
WALL_CIRCLE = _REGISTRY["wall"]["circle"]

__all__ = ["Route", "resolve", "routes_of", "route_manifest"]
