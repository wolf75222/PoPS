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
tokens, same order); ``tests/python/architecture/test_route_registry_parity.py`` locks the two at the
source level, and the C++ static_asserts lock route_ids.hpp against the historical tag tables
(kLimiters / kRiemanns / kTransports / kSources / kElliptics). Deliberately IMPORT-FREE (stdlib
only): the architecture gate loads it standalone, without the compiled ``_pops`` module.
"""
from __future__ import annotations

from typing import Any


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

    def __new__(cls, family: str, token: str, native_entry: str,
                requirements: Any = (), limitations: Any = ()) -> Route:
        self = super().__new__(cls, token)
        self.family = family
        self.id = "%s.%s" % (family, token)
        self.native_entry = native_entry
        self.requirements = tuple(requirements)
        self.limitations = tuple(limitations)
        return self

    @property
    def token(self) -> str:
        """The legacy wire token (the ``str`` value itself; debug / ABI / messages only)."""
        return str(self)

    def manifest(self) -> dict:
        """The structured manifest row of this route (inspection / reports)."""
        return {
            "family": self.family,
            "id": self.id,
            "token": str(self),
            "native_entry": self.native_entry,
            "requirements": list(self.requirements),
            "limitations": list(self.limitations),
        }

    def __repr__(self) -> str:
        return "Route(%s)" % self.id


def _split(csv: str) -> tuple:
    return tuple(s for s in csv.split(",") if s) if csv else ()


# --- The route tables: MIRROR of route_ids.hpp, one row per line, same order. ------------------
# (family, ((token, native_entry, requirements_csv, limitations_csv), ...))
_TABLES = {
    "riemann": (
        ("rusanov", "pops::RusanovFlux", "max_wave_speed", ""),
        ("hll", "pops::HLLFlux", "physical_flux,wave_speeds", ""),
        ("hllc", "pops::HLLCFlux",
         "physical_flux,pressure,wave_speeds,contact_speed,hllc_star_state",
         "polar geometry not wired; generic-only (ADC-590), requires HasHLLCStructure"),
        ("roe", "pops::RoeFlux", "physical_flux,roe_average",
         "polar geometry not wired; generic-only (ADC-590), requires HasRoeDissipation"),
        ("euler_hllc", "pops::EulerHLLCFlux2D", "physical_flux,pressure,euler_2d_layout",
         "4-variable canonical Euler (rho,mx,my,E) only; explicit route, never a fallback; "
         "polar not wired"),
        ("euler_roe", "pops::EulerRoeFlux2D", "physical_flux,pressure,euler_2d_layout",
         "4-variable canonical Euler (rho,mx,my,E) only; explicit route, never a fallback; "
         "polar not wired"),
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


def resolve(family: str, token: str, context: str = "routes") -> Route:
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


def routes_of(family: str) -> tuple:
    """The ordered typed routes of @p family (registry order = route_ids.hpp order)."""
    return tuple(_REGISTRY[family].values())


def route_manifest() -> list:
    """The full structured route manifest (every family, registry order) -- inspection surface."""
    return [route.manifest() for family in _TABLES for route in _REGISTRY[family].values()]


# Native route catalog version (ADC-599): bumped on any INCOMPATIBLE registry change (a removed
# or re-tokenized route). Additive rows do not need a bump -- the registry hash below already
# separates artifacts built against different route sets.
ROUTE_REGISTRY_VERSION = 1

# Capabilities/reports VOCABULARY version (ADC-599): the shared vocabulary of the capability and
# inspection reports (status tokens available/partial/unavailable, route row fields, manifest
# keys). Enters every compiled-artifact cache key so an artifact whose embedded reports speak an
# older vocabulary is not silently reused after an incompatible vocabulary change.
CAPABILITY_VOCAB_VERSION = 0


def route_registry_signature() -> str:
    """Compact per-family signature "family:count,..." (registry order) -- the embedded form.

    MIRROR of pops::route_registry_signature() (route_ids.hpp); the two strings must stay equal
    (locked by tests/python/architecture/test_route_registry_parity.py). Embedded verbatim in generated
    artifacts so a stale .so is refused with the mismatching family named, before any run.
    """
    return ",".join("%s:%d" % (family, len(_TABLES[family])) for family in _TABLES)


def route_registry_hash() -> str:
    """Stable hash of the FULL route registry (tokens, entries, requirements, limitations).

    Enters every compiled-artifact cache key (ADC-599): any registry change -- a new route, a
    renamed native entry, an added limitation -- invalidates cached .so files instead of silently
    reusing an artifact built against a different native catalog.
    """
    import hashlib
    parts = ["v%d" % ROUTE_REGISTRY_VERSION]
    for family in _TABLES:
        for token, entry, req, lim in _TABLES[family]:
            parts.append("%s|%s|%s|%s|%s" % (family, token, entry, req, lim))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


# --- Typed route constants (the internal currency of the lowering layer) -----------------------
RIEMANN_RUSANOV = _REGISTRY["riemann"]["rusanov"]
RIEMANN_HLL = _REGISTRY["riemann"]["hll"]
RIEMANN_HLLC = _REGISTRY["riemann"]["hllc"]
RIEMANN_ROE = _REGISTRY["riemann"]["roe"]
RIEMANN_EULER_HLLC = _REGISTRY["riemann"]["euler_hllc"]
RIEMANN_EULER_ROE = _REGISTRY["riemann"]["euler_roe"]

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

def euler_layout_ok(compiled: Any, flux: Any) -> bool:
    """True when @p compiled is a canonical 4-variable Euler transport (n_vars == 4 + primitive 'p')
    that did NOT emit the generic capability for @p flux -- the acceptance test for the explicit
    euler_hllc / euler_roe routes (ADC-590). Shared by the System and unified install guards."""
    emitted = getattr(compiled, "has_hllc" if flux in ("euler_hllc", "hllc") else "has_roe", False)
    return (getattr(compiled, "n_vars", 0) == 4
            and "p" in getattr(compiled, "prim_names", []) and not emitted)


def check_riemann_capability(flux: Any, compiled: Any, ctx: Any) -> None:
    """Gate the selected Riemann flux against the model's emitted capabilities (ADC-590).

    Shared by System.add_equation and AmrSystem.add_equation (@p flux is a Route or a bare wire
    token; both compare equal to the token string). Generic hllc/roe are GENERIC-ONLY now: the
    model MUST carry the capability (``has_hllc`` / ``has_roe``). The canonical 4-variable Euler
    layout is served by the EXPLICIT euler_hllc / euler_roe routes, which require n_vars == 4 +
    primitive 'p' and REFUSE a model that emitted the generic capability (no ambiguity). Raises
    ``ValueError`` with a @p ctx-prefixed message that names the missing capability and both
    remedies. HLL keeps its own wave-speeds guard at the call-site; the ADC-552 provider cross-check
    rides through :func:`pops.numerics.riemann.waves.check_hll_waves` at the call site (routes.py
    stays import-free of the pops package).
    """
    def _tail() -> str:
        return ("[requested route %s -> %s; requires: %s]"
                % (getattr(flux, "id", flux), getattr(flux, "native_entry", "?"),
                   ", ".join(getattr(flux, "requirements", ()))))
    if ((flux == "hllc" and not getattr(compiled, "has_hllc", False))
            or (flux == "roe" and not getattr(compiled, "has_roe", False))):
        cap = "hllc_star_state" if flux == "hllc" else "roe_dissipation"
        enable = "m.enable_hllc()" if flux == "hllc" else "m.enable_roe()"
        euler = "EulerHLLC2D()" if flux == "hllc" else "EulerRoe2D()"
        raise ValueError(
            "%s: riemann '%s' requires the model capability '%s': call %s on a generic model "
            "(roles + primitive 'p'), or select the explicit canonical Euler route riemann=%s for "
            "a 4-variable Euler (rho,rho_u,rho_v,E) transport; otherwise use riemann='rusanov' %s"
            % (ctx, flux, cap, enable, euler, _tail()))
    if flux in ("euler_hllc", "euler_roe") and not euler_layout_ok(compiled, flux):
        generic = "hllc" if flux == "euler_hllc" else "roe"
        raise ValueError(
            "%s: riemann '%s' requires a canonical 4-variable Euler transport (n_vars == 4, "
            "primitive 'p', layout rho/rho_u/rho_v/E) and NO emitted generic capability; for a "
            "generic model that called m.enable_hllc()/m.enable_roe() use riemann='%s' instead; "
            "for a non-Euler model use riemann='rusanov'/'hll' %s"
            % (ctx, flux, generic, _tail()))


def check_wave_speed_provider(requested_kind: Any, compiled: Any, ctx: Any,
                              actual_provider: Any = None) -> None:
    """Cross-check an HLL(waves=<provider>) request against the compiled model's source (ADC-552).

    @p requested_kind is the provider kind an ``HLL(waves=...)`` descriptor pinned (a signed-pair
    kind: ``explicit_pair`` / ``jacobian`` / ``pressure_derived`` / ``einfeldt`` / ``davis``). The
    model MUST at least emit wave speeds (``has_wave_speeds``): a mismatch is refused with a message
    naming the requested provider and the model's actual provider. @p actual_provider is the model's
    DERIVED source kind (``explicit_pair`` / ``jacobian`` / ``pressure_derived``) or ``None`` -- the
    caller computes it via :func:`pops.numerics.riemann.waves.provider_of` (routes.py stays
    import-free of the pops package). The estimate kinds (``einfeldt`` / ``davis``) are compatible
    with any signed source. When @p actual_provider is ``None`` (a bare CompiledModel whose source
    kind is unrecorded) the request is ACCEPTED once ``has_wave_speeds`` is True (a documented,
    honest limitation -- the ``.so`` metadata does not carry the source kind).
    """
    if not getattr(compiled, "has_wave_speeds", True):
        raise ValueError(
            "%s: riemann 'hll' with a wave-speed provider %r requires the model to emit signed "
            "wave speeds, but it emits none; declare m.wave_speeds(x=(smin, smax), y=(smin, smax)) "
            "or m.wave_speeds_from_jacobian(...) or a primitive 'p', or use riemann='rusanov'."
            % (ctx, requested_kind))
    if requested_kind in ("einfeldt", "davis"):
        return  # estimate providers are compatible with any signed wave-speed source
    if actual_provider not in ("explicit_pair", "jacobian", "pressure_derived"):
        return  # source kind not derivable on this handle: accept (documented limitation)
    if actual_provider != requested_kind:
        raise ValueError(
            "%s: riemann 'hll' was pinned to the wave-speed provider %r, but the model's actual "
            "wave-speed source is %r; pass HLL(waves=%s) matching the model, or declare the "
            "requested source." % (ctx, requested_kind, actual_provider,
                                    _provider_factory(actual_provider)))


def _provider_factory(kind: Any) -> str:
    """The typed factory name for a provider @p kind (used in the mismatch message)."""
    return {"explicit_pair": "ExplicitPair()", "jacobian": "FromJacobian()",
            "pressure_derived": "FromPressure()", "einfeldt": "Einfeldt()",
            "davis": "Davis()"}.get(kind, "the matching provider")


__all__ = ["Route", "resolve", "routes_of", "route_manifest", "check_riemann_capability",
           "check_wave_speed_provider", "euler_layout_ok"]
