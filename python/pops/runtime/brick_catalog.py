"""Builtin native brick catalog (ADC-586): the Python mirror of brick_catalog.hpp.

The catalog is the single inspectable surface of the bricks the core ships: ONE declarative row
per canonical model brick (3 transports + 5 canonical sources + 3 elliptics), carrying the id,
category, typed route index, native C++ entry point, the CSV of :class:`ModelSpec` constructor
params, the variable-count / polar capability, the requirements / capabilities contract and a
one-line summary. It answers "what native bricks exist, with which identity and construction
contract" without touching the compiled ``_pops`` extension.

This module is the MIRROR of ``include/pops/runtime/builders/factory/brick_catalog.hpp`` (same
categories, same ordered ids, same native entries / params / n_vars / polar_ok); the C++ header
static_asserts itself against the registry (model_registry.hpp) and route (route_ids.hpp) tables,
and ``tests/python/architecture/test_brick_catalog_parity.py`` locks this table against the header at the
source level (no build). Deliberately IMPORT-FREE (stdlib only, like ``pops.runtime.routes``): the
architecture gate loads it standalone, before the compiled ``_pops`` module exists.

The alias source spellings (``lorentz`` / ``potential_lorentz``) are PARSE-ONLY in
``pops.runtime.routes`` (``_ALIASES``); they are never catalog rows, so each catalog entry is one
canonical brick.
"""

# --- The catalog table: MIRROR of kBrickCatalog (brick_catalog.hpp), one row per line, same order.
# (category, id, route_index, native_entry, params_csv, n_vars, polar_ok, requirements_csv,
#  capabilities_csv, summary)
_TABLE = (
    ("transport", "exb", 0, "pops::ExBVelocity", "B0", 1, True, "",
     "scalar (1 var); no fluid source",
     "scalar ExB drift advection, v = (-d_y phi, d_x phi) / B0"),
    ("transport", "compressible", 1, "pops::CompressibleFlux", "gamma", 4, False, "",
     "polar geometry not wired", "compressible Euler, 4 var (rho, rho u, rho v, E)"),
    ("transport", "isothermal", 2, "pops::IsothermalFlux", "cs2,vacuum_floor", 3, True, "", "",
     "isothermal Euler, 3 var (rho, rho u, rho v)"),
    ("source", "none", 0, "pops::NoSource", "", 1, False, "", "", "neutral: no source term"),
    ("source", "potential", 1, "pops::PotentialForce", "qom", 3, False,
     "fluid transport (>= 3 vars)", "", "(q/m) rho E electrostatic force"),
    ("source", "gravity", 2, "pops::GravityForce", "", 3, False, "fluid transport (>= 3 vars)", "",
     "rho g gravity force"),
    ("source", "magnetic", 3, "pops::MagneticLorentzForce", "qom", 3, False,
     "fluid transport (>= 3 vars),aux B_z channel",
     "explicit regime (stiff regime -> condensed Schur stage)",
     "q v x B_z magnetized Lorentz force (explicit regime)"),
    ("source", "potential_magnetic", 4,
     "pops::CompositeSource<PotentialForce, MagneticLorentzForce>", "qom", 3, False,
     "fluid transport (>= 3 vars),aux B_z channel", "", "electrostatic + Lorentz, summed"),
    ("elliptic", "charge", 0, "pops::ChargeDensity", "q", -1, False, "", "",
     "rho - q : charge density (Poisson source)"),
    ("elliptic", "background", 1, "pops::BackgroundDensity", "alpha,n0", -1, False, "", "",
     "alpha (rho - n0) : neutralizing background"),
    ("elliptic", "gravity", 2, "pops::GravityCoupling", "sign,four_pi_G,rho0", -1, False, "", "",
     "sign * 4 pi G (rho - rho0) : gravitational coupling"),
)

_FIELDS = ("category", "id", "route_index", "native_entry", "params", "n_vars", "polar_ok",
           "requirements", "capabilities", "summary")


def _row_dict(row):
    """One catalog row as a plain dict; the CSV columns become lists (JSON-ready)."""
    entry = dict(zip(_FIELDS, row, strict=True))
    entry["params"] = [p for p in entry["params"].split(",") if p]
    entry["requirements"] = [p for p in entry["requirements"].split(",") if p]
    entry["capabilities"] = [p for p in entry["capabilities"].split(",") if p]
    return entry


def brick_catalog():
    """The full builtin brick catalog: an ordered list of dicts (one per canonical brick).

    Registry order (3 transports, 5 canonical sources, 3 elliptics). Each dict carries the id,
    category, route_index, native_entry, params, n_vars, polar_ok, requirements, capabilities and
    summary -- the inspection surface the codegen manifest and the reports read.
    """
    return [_row_dict(row) for row in _TABLE]


def catalog_ids(category):
    """The ordered canonical ids of @p category (mirror of C++ catalog_csv, as a list)."""
    return [row[1] for row in _TABLE if row[0] == category]


def resolve(category, id, context="brick catalog"):
    """Resolve (@p category, @p id) to its catalog entry -- refuse an unknown one, never default.

    The refusal lists the catalog entries of that category (derived from the table, never
    hand-written), so a typo is diagnosable from the catalog alone (ADC-586: errors list the
    available catalog entries, not a hand-coded token list).
    """
    for row in _TABLE:
        if row[0] == category and row[1] == id:
            return _row_dict(row)
    ids = catalog_ids(category)
    if not ids:
        known_categories = []
        for row in _TABLE:
            if row[0] not in known_categories:
                known_categories.append(row[0])
        raise ValueError("%s: unknown category %r (valid: %s)"
                         % (context, category, "|".join(known_categories)))
    raise ValueError("%s: unknown %s brick %r (catalog: %s); the catalog never falls back "
                     "to a default" % (context, category, id, "|".join(ids)))


__all__ = ["brick_catalog", "catalog_ids", "resolve"]
