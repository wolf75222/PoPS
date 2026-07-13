"""Typed native route API over the generated schema-v2 component catalog.

Every algorithmic choice (Riemann flux, limiter, reconstructed variables, time treatment,
splitting, field solver, Poisson boundary condition, layout, model bricks) is identified by a
typed :class:`Route`, not by a free string. The typed descriptors (``pops.numerics`` /
``pops.solvers`` / the time bricks) lower to these Routes; the ONLY places that emit the legacy
wire token toward the C++ ABI are the bounded adapters (``pops.runtime._system_install`` and
``pops.codegen.compile_emit``), and they emit ``str(route)`` -- a :class:`Route` IS its wire
token (``str`` subclass), so the crossing stays byte-identical while the identity, requirements,
limitations and native entry point become typed and inspectable.

All route declarations, wire IDs, aliases, capabilities and entry points come from
``schemas/component_catalog.v2.json``. Checked-in Python and C++ products are generated together;
this module contains behavior only and cannot drift into a second registry.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any

from ._generated_component_routes import (
    CAPABILITY_VOCAB_VERSION,
    COMPONENT_CATALOG_SEMANTIC_SHA256,
    COMPONENT_CATALOG_SHA256,
    COMPONENT_MANIFEST_SCHEMA_VERSION,
    ROUTE_ALIASES as _GENERATED_ROUTE_ALIASES,
    ROUTE_COMPONENT_DEFAULTS as _GENERATED_COMPONENT_DEFAULTS,
    ROUTE_METADATA as _GENERATED_ROUTE_METADATA,
    ROUTE_REGISTRY_SIGNATURE,
    ROUTE_REGISTRY_VERSION,
    ROUTE_TABLES as _GENERATED_ROUTE_TABLES,
)


def _freeze_catalog_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_catalog_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_catalog_value(item) for item in value)
    return value


def _thaw_catalog_value(value: Any) -> Any:
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _thaw_catalog_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_catalog_value(item) for item in value]
    return value


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
    metadata: dict

    def __new__(cls, family: str, token: str, native_entry: str,
                requirements: Any = (), limitations: Any = (), metadata: Any = None) -> Route:
        self = super().__new__(cls, token)
        self.family = family
        self.id = "%s.%s" % (family, token)
        self.native_entry = native_entry
        self.requirements = tuple(requirements)
        self.limitations = tuple(limitations)
        self.metadata = _freeze_catalog_value(dict(metadata or {}))
        self._frozen = True
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("Route values are immutable generated catalog handles")
        object.__setattr__(self, name, value)

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
            "capabilities": _thaw_catalog_value(self.metadata),
            "catalog_digest": COMPONENT_CATALOG_SHA256,
            "catalog_semantic_digest": COMPONENT_CATALOG_SEMANTIC_SHA256,
        }

    def component_contract(self) -> dict:
        """Complete schema-v2 contract input for registration as a component."""
        defaults = _COMPONENT_DEFAULTS
        metadata = _thaw_catalog_value(self.metadata)
        summary = metadata.pop("summary", "")
        parameters = list(defaults["parameters"])
        parameters.extend(metadata.pop("parameters", ()))
        return {
            "schema_version": COMPONENT_MANIFEST_SCHEMA_VERSION,
            "uri": "pops://builtin/routes/%s/%s" % (self.family, self.token),
            "component_type": "route.%s" % self.family,
            "version": dict(defaults["version"]),
            "facets": list(defaults["facets"]),
            "signature": dict(defaults["signature"]),
            "reads": list(defaults["reads"]),
            "writes": list(defaults["writes"]),
            "parameters": parameters,
            "interfaces": list(defaults["interfaces"]),
            "requirements": list(self.requirements),
            "capabilities": [{"name": key, "value": value}
                             for key, value in sorted(metadata.items())],
            "effects": list(defaults["effects"]),
            "layouts": list(defaults["layouts"]),
            "clocks": list(defaults["clocks"]),
            "target": _thaw_catalog_value(defaults["target"]),
            "determinism": dict(defaults["determinism"]),
            "restart": dict(defaults["restart"]),
            "precision": {key: list(value) if isinstance(value, list) else value
                          for key, value in defaults["precision"].items()},
            "conservation": list(defaults["conservation"]),
            "entry_points": {"native": self.native_entry},
            "extensions": {
                "pops://schemas/extensions/route-inspection": {
                    "kind": "documentary",
                    "data": {
                        "family": self.family,
                        "id": self.id,
                        "token": str(self),
                        "summary": summary,
                        "limitations": list(self.limitations),
                        "catalog_digest": COMPONENT_CATALOG_SHA256,
                        "catalog_semantic_digest": COMPONENT_CATALOG_SEMANTIC_SHA256,
                    },
                },
            },
        }

    def component_manifest(self):
        """Materialize the complete canonical ComponentManifest lazily."""
        from pops.model import ComponentManifest

        return ComponentManifest(**self.component_contract())

    def __repr__(self) -> str:
        return "Route(%s)" % self.id


_TABLES = _freeze_catalog_value(_GENERATED_ROUTE_TABLES)
_ALIASES = _freeze_catalog_value(_GENERATED_ROUTE_ALIASES)
_COMPONENT_DEFAULTS = _freeze_catalog_value(_GENERATED_COMPONENT_DEFAULTS)


_REGISTRY = {
    family: {token: Route(family, token, entry, req, lim,
                          _GENERATED_ROUTE_METADATA[family][token])
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


def component_manifests() -> tuple:
    """Canonical schema-v2 manifests for all builtin route components."""
    return tuple(route.component_manifest()
                 for family in _TABLES for route in _REGISTRY[family].values())


def route_registry_signature() -> str:
    """Generated content identity shared verbatim with the C++ catalog."""
    return ROUTE_REGISTRY_SIGNATURE


def route_registry_hash() -> str:
    """Stable hash of the behavior-bearing route registry.

    Enters every compiled-artifact cache key (ADC-599): any registry change -- a new route, a
    renamed native entry, a changed requirement or capability -- invalidates cached .so files
    instead of silently reusing an artifact built against a different native catalog. Documentary
    summaries and limitation prose belong only to the full catalog digest and do not recompile.
    """
    return COMPONENT_CATALOG_SEMANTIC_SHA256


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

# Named route-id constants exist ONLY for identifiers with a live consumer (ADC-630: the
# symmetric-per-family constant surface was dead code and was deleted; read _REGISTRY["<family>"]
# ["<name>"] directly where a one-off id is needed).
SOURCE_MAGNETIC = _REGISTRY["source"]["magnetic"]

SOURCE_STAGE_ELECTROSTATIC_LORENTZ = _REGISTRY["source_stage"]["electrostatic_lorentz"]

def euler_layout_ok(compiled: Any, flux: Any) -> bool:
    """True when @p compiled is a canonical 4-variable Euler transport (n_vars == 4 + primitive 'p')
    that did NOT emit the generic capability for @p flux -- the acceptance test for the explicit
    euler_hllc / euler_roe routes (ADC-590). Shared by the System and unified install guards."""
    emitted = getattr(compiled, "has_hllc" if flux in ("euler_hllc", "hllc") else "has_roe", False)
    return (getattr(compiled, "n_vars", 0) == 4
            and "p" in getattr(compiled, "prim_names", []) and not emitted)


# ADC-642: the ONE per-flux capability-gate catalog. A generic capability-backed Riemann flux is
# one row {token: (model_flag, capability_token, enable_hint, euler_route)}; the explicit canonical-
# Euler routes map to their generic counterpart. check_riemann_capability, the System / AMR / unified
# install guards and pops.numerics.riemann.availability all read THIS -- no per-flux branch re-listing.
_RIEMANN_CAPABILITY_GATES = {
    "hllc": ("has_hllc", "hllc_star_state", "m.enable_hllc()", "EulerHLLC2D()"),
    "roe": ("has_roe", "roe_dissipation", "m.enable_roe()", "EulerRoe2D()"),
}
_EULER_ROUTE_GENERIC = {"euler_hllc": "hllc", "euler_roe": "roe"}


def check_riemann_capability(flux: Any, compiled: Any, ctx: Any) -> None:
    """Gate the selected Riemann flux against the model's emitted capabilities (ADC-590).

    Shared by System.add_equation and AmrSystem.add_equation (@p flux is a Route or a bare wire
    token; both compare equal to the token string). Generic hllc/roe are GENERIC-ONLY now: the
    model MUST carry the capability (``has_hllc`` / ``has_roe``). The canonical 4-variable Euler
    layout is served by the EXPLICIT euler_hllc / euler_roe routes, which require n_vars == 4 +
    primitive 'p' and REFUSE a model that emitted the generic capability (no ambiguity). Raises
    ``ValueError`` with a @p ctx-prefixed message that names the missing capability and both
    remedies. Reads the ADC-642 :data:`_RIEMANN_CAPABILITY_GATES` catalog (one row per flux). HLL
    keeps its own wave-speeds guard at the call-site; the ADC-552 provider cross-check rides through
    :func:`pops.numerics.riemann.waves.check_hll_waves` at the call site (routes.py stays import-free
    of the pops package).
    """
    def _tail() -> str:
        return ("[requested route %s -> %s; requires: %s]"
                % (getattr(flux, "id", flux), getattr(flux, "native_entry", "?"),
                   ", ".join(getattr(flux, "requirements", ()))))
    gate = _RIEMANN_CAPABILITY_GATES.get(flux)
    if gate is not None:
        model_flag, cap, enable, euler = gate
        if not getattr(compiled, model_flag, False):
            raise ValueError(
                "%s: riemann '%s' requires the model capability '%s': call %s on a generic model "
                "(roles + primitive 'p'), or select the explicit canonical Euler route riemann=%s "
                "for a 4-variable Euler (rho,rho_u,rho_v,E) transport; otherwise use "
                "riemann='rusanov' %s" % (ctx, flux, cap, enable, euler, _tail()))
    if flux in _EULER_ROUTE_GENERIC and not euler_layout_ok(compiled, flux):
        generic = _EULER_ROUTE_GENERIC[flux]
        raise ValueError(
            "%s: riemann '%s' requires a canonical 4-variable Euler transport (n_vars == 4, "
            "primitive 'p', layout rho/rho_u/rho_v/E) and NO emitted generic capability; for a "
            "generic model that called m.enable_hllc()/m.enable_roe() use riemann='%s' instead; "
            "for a non-Euler model use riemann='rusanov'/'hll' %s"
            % (ctx, flux, generic, _tail()))


def riemann_missing_capabilities(flux) -> list:
    """The capability token(s) a model must emit for @p flux (report / availability surface).

    Reads the ADC-642 :data:`_RIEMANN_CAPABILITY_GATES` catalog; the availability report
    (:func:`pops.numerics.riemann.availability.flux_available`) reads this instead of re-listing the
    per-flux capability strings.
    """
    gate = _RIEMANN_CAPABILITY_GATES.get(flux)
    if gate is not None:
        return [gate[1]]
    if flux in _EULER_ROUTE_GENERIC:
        return ["euler_2d_layout"]
    if flux == "hll":
        return ["wave_speeds"]
    return []


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


__all__ = ["Route", "resolve", "routes_of", "route_manifest", "component_manifests",
           "check_riemann_capability",
           "check_wave_speed_provider", "euler_layout_ok", "riemann_missing_capabilities"]
