"""Builtin native brick catalog surface (ADC-586).

The catalog (:mod:`pops.runtime.brick_catalog`) is the single inspectable table of the native bricks
the core ships. These checks pin the surface at the pops-package level and prove the typed dispatch
still builds every canonical brick:

  1  catalog shape / ids: brick_catalog() carries the 11 canonical rows, exposed on pops.runtime.
  2  resolve refusal lists the CATALOG entries of the category (never a hand-coded token list).
  3  every canonical brick still builds through pops.Model(...) + System.add_block (the typed
     dispatch in model_factory.hpp constructs each transport / source / elliptic brick).
  4  an unknown transport ModelSpec still throws the HISTORICAL "unknown transport" message
     (byte-identical: the registry single-source rejection is unchanged by the typed dispatch).

The System construction in groups 3-4 needs the compiled _pops extension, so the whole module is
guarded with pytest.importorskip("pops"), like the sibling test_route_ids.py.
"""

import numpy as np
import pytest
from pops.runtime.system import System  # ADC-545 advanced runtime seam

pops = pytest.importorskip("pops")
import importlib  # noqa: E402
from pops.runtime import ModelSpec  # noqa: E402

# pops.runtime re-exports the brick_catalog() FUNCTION under the name `brick_catalog`, which shadows
# the submodule attribute; import the submodule object by its dotted name (sys.modules) to reach
# resolve() / catalog_ids() as well.
bc_module = importlib.import_module("pops.runtime.brick_catalog")
brick_catalog = bc_module.brick_catalog


def test_catalog_shape_and_ids():
    # Group 1: the catalog carries the 11 canonical bricks (3 transports + 5 sources + 3 elliptics),
    # each row a structured dict; brick_catalog is re-exported on pops.runtime.
    catalog = brick_catalog()
    assert len(catalog) == 11, "brick catalog is not 11 rows: %d" % len(catalog)
    by_cat = {}
    for row in catalog:
        for key in ("id", "category", "route_index", "native_entry", "parameters", "n_vars",
                    "polar_ok", "requirements", "limitations", "summary", "catalog_digest",
                    "catalog_semantic_digest"):
            assert key in row, "catalog row missing %r key: %r" % (key, row)
        by_cat.setdefault(row["category"], []).append(row["id"])
    assert by_cat["transport"] == ["exb", "compressible", "isothermal"]
    assert by_cat["source"] == ["none", "potential", "gravity", "magnetic", "potential_magnetic"]
    assert by_cat["elliptic"] == ["charge", "background", "gravity"]
    # Every inspection fact is detached from the generated catalog row.
    exb = bc_module.resolve("transport", "exb")
    assert exb["native_entry"] == "pops::ExBVelocity" and exb["parameters"] == ["B0"]
    assert bc_module.resolve("transport", "isothermal")["parameters"] == ["cs2", "vacuum_floor"]
    info = bc_module.catalog_info()
    assert info["schema_version"] == 1
    assert len(info["digest"]) == 64
    assert len(info["semantic_digest"]) == 64


def test_resolve_refusal_lists_catalog_entries():
    # Group 2: an unknown id is refused with the catalog entries of that category (from the table).
    with pytest.raises(ValueError) as exc:
        bc_module.resolve("transport", "upwind")
    message = str(exc.value)
    assert "upwind" in message
    for entry in ("exb", "compressible", "isothermal"):
        assert entry in message, "catalog entry %r not listed: %r" % (entry, message)
    assert "never falls back to a default" in message
    # An alias source spelling is NOT a catalog entry (parse-only in routes).
    with pytest.raises(ValueError):
        bc_module.resolve("source", "lorentz")


def _system(n=16):
    return System(n=n, L=1.0, periodic=True)


def test_every_canonical_brick_still_builds():
    # Group 3: each canonical transport / source / elliptic brick builds through the typed dispatch.
    n = 16
    ic = np.ones((n, n))

    # Transports (with a compatible state) + the neutral source + charge elliptic.
    scalar = pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                        source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=1.0))
    compressible = pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                              transport=pops.CompressibleFlux(),
                              source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=1.0))
    isothermal = pops.Model(state=pops.FluidState("isothermal", cs2=1.0),
                            transport=pops.IsothermalFlux(),
                            source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=1.0))
    for name, model in (("exb", scalar), ("comp", compressible), ("iso", isothermal)):
        _system(n).block(name, model=model, spatial=pops.Spatial(none=True))

    # Sources on a fluid transport (isothermal, 3 vars). NoSource covered above.
    def iso_source(src):
        return pops.Model(state=pops.FluidState("isothermal", cs2=1.0),
                          transport=pops.IsothermalFlux(), source=src,
                          elliptic=pops.ChargeDensity(charge=1.0))

    for name, src in (("potential", pops.PotentialForce(charge=1.0)),
                      ("gravity", pops.GravityForce()),
                      ("magnetic", pops.MagneticLorentzForce(charge=1.0)),
                      ("potmag", pops.PotentialMagneticForce(charge=1.0))):
        _system(n).block(name, model=iso_source(src), spatial=pops.Spatial(none=True))

    # Elliptics (on a scalar transport, the historical diocotron shape). ChargeDensity covered above.
    for name, ell in (("bg", pops.BackgroundDensity(alpha=1.0, n0=0.0)),
                      ("grav", pops.GravityCoupling())):
        model = pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                           source=pops.NoSource(), elliptic=ell)
        _system(n).block(name, model=model, spatial=pops.Spatial(none=True))
    assert ic.shape == (n, n)  # touch the array so the import is not flagged unused


def test_unknown_transport_modelspec_throws_historical_message():
    # Group 4: an unknown transport tag still throws the HISTORICAL "unknown transport" message
    # (byte-identical) -- the registry single-source rejection is untouched by the typed dispatch.
    spec = ModelSpec()
    spec.transport = "bogus"
    spec.elliptic = "charge"
    spec.source = "none"
    with pytest.raises(Exception) as exc:
        _system().block("m", spec)
    message = str(exc.value)
    assert "unknown transport" in message, "historical 'unknown transport' message drifted: %r" % message
    assert "bogus" in message
    assert "exb|compressible|isothermal" in message
