"""Typed native route layer: registry, lowering and pre-bind refusal (ADC-584).

The typed-route layer (:mod:`pops.runtime.routes`) gives every algorithmic choice a typed
:class:`~pops.runtime.routes.Route` (a ``str`` subclass) instead of a free wire string. These
checks pin the layer at the pure pops-package level, without stepping a System or compiling a
.so:

  1  registry integrity: route_manifest() shape, unique ids, token == str value.
  2  typed lowering: Spatial() defaults carry Route objects that still str-equal the historical
     tokens (minmod / rusanov / conservative).
  3  typed descriptors lower to routes: Spatial(flux=HLL()) -> riemann.hll.
  4  explicit / IMEX time treatments expose their typed time route.
  5  an unknown route is refused, never defaulted (resolve raises, listing the valid set).
  6  historical alias spellings resolve to their canonical route.
  7  the routes() inspection surface reports the chosen routes and their limitations.
  8  set_poisson pre-validates every behavior token BEFORE the C++ boundary.
  9  the external-flux "user" token stays a plain token (no native route).

The System construction in group 8 needs the compiled _pops extension, so the whole module is
guarded with pytest.importorskip("pops"), exactly like the sibling
test_runtime_inspection_reports.py.
"""
import sys

import pytest

pops = pytest.importorskip("pops")
from pops.numerics.riemann import HLL  # noqa: E402
from pops.runtime import routes  # noqa: E402
from pops.runtime._bricks_scheme import _FLUX_SCHEMES  # noqa: E402


def test_route_manifest_registry_integrity():
    # Group 1: the manifest is the full inspection surface. Every row is a structured route, the
    # ids are unique, and the token IS the route's str value (id == "family.token").
    manifest = routes.route_manifest()
    assert len(manifest) >= 40, "route registry shrank below 40 entries: %d" % len(manifest)
    seen = set()
    for row in manifest:
        for key in ("family", "id", "token", "native_entry"):
            assert key in row, "manifest row missing %r key: %r" % (key, row)
        route = routes.resolve(row["family"], row["token"])
        assert str(route) == row["token"], "token %r != str value %r" % (row["token"], str(route))
        assert route.id == row["id"] == "%s.%s" % (row["family"], row["token"]), row
        assert row["id"] not in seen, "duplicate route id %r" % row["id"]
        seen.add(row["id"])


def test_spatial_defaults_lower_to_typed_routes():
    # Group 2: the Spatial defaults are typed Routes whose str value stays the historical token.
    spatial = pops.Spatial()
    assert isinstance(spatial.limiter, routes.Route)
    assert spatial.limiter.id == "limiter.minmod"
    assert spatial.flux.id == "riemann.rusanov"
    assert spatial.recon.id == "recon.conservative"
    # A Route IS its wire token: the historical string comparisons stay byte-identical.
    assert spatial.limiter == "minmod"
    assert spatial.flux == "rusanov"
    assert spatial.recon == "conservative"


def test_typed_descriptor_lowers_to_route():
    # Group 3: a typed pops.numerics descriptor lowers to its native route + entry point.
    spatial = pops.Spatial(flux=HLL())
    assert spatial.flux.id == "riemann.hll"
    assert spatial.flux.native_entry == "pops::HLLFlux"


def test_time_treatments_expose_typed_route():
    # Group 4: the explicit / IMEX time treatments carry a typed time route on .kind.
    assert pops.Explicit().kind.id == "time.explicit"
    assert pops.Explicit(ssprk3=True).kind.id == "time.ssprk3"
    assert pops.IMEX().kind.id == "time.imex"


def test_unknown_route_is_refused_never_defaulted():
    # Group 5: an unknown route raises and cites the valid set; it never falls back to a default.
    with pytest.raises(ValueError, match="never fall back to a default") as exc:
        routes.resolve("riemann", "upwind")
    message = str(exc.value)
    assert "upwind" in message
    for valid in ("rusanov", "hll", "hllc", "roe", "euler_hllc", "euler_roe"):
        assert valid in message, "valid route %r not listed: %r" % (valid, message)


def test_unknown_family_names_the_valid_families():
    # Group 5 (cont.): an unknown family raises and names the valid families.
    with pytest.raises(ValueError) as exc:
        routes.resolve("bogus_family", "x")
    message = str(exc.value)
    assert "bogus_family" in message
    for family in ("riemann", "field_solver", "source"):
        assert family in message, "family %r not listed: %r" % (family, message)


def test_alias_spellings_resolve_to_canonical_route():
    # Group 6: historical alias spellings resolve to the canonical route object (identity).
    assert routes.resolve("source", "lorentz") is routes.SOURCE_MAGNETIC
    assert routes.resolve("time", "ssprk2") is routes.TIME_EXPLICIT


def test_routes_inspection_surface():
    # Group 7: the routes() inspection surface reports the chosen routes and their limitations.
    scheme = pops.Spatial(weno5=True).routes()
    assert set(scheme) == {"limiter", "riemann", "recon"}
    assert scheme["limiter"]["id"] == "limiter.weno5"
    assert scheme["limiter"]["limitations"], "weno5 declares a native limitation"
    assert pops.Explicit().routes()["time"]["id"] == "time.explicit"


def test_set_poisson_refuses_unknown_routes_before_bind():
    # Group 8: set_poisson resolves every behavior token to its typed route BEFORE the C++
    # boundary, so an unknown token is refused (with the family + the token) and never binds.
    def system():
        return pops.System(n=8, L=1.0, periodic=True)

    with pytest.raises(ValueError, match="field_solver") as exc:
        system().set_poisson(solver="bogus_solver")
    assert "bogus_solver" in str(exc.value)
    with pytest.raises(ValueError, match="poisson_bc"):
        system().set_poisson(bc="bogus")
    with pytest.raises(ValueError, match="poisson_rhs"):
        system().set_poisson(rhs="bogus")
    with pytest.raises(ValueError, match="wall"):
        system().set_poisson(wall="bogus")
    # A fully valid configuration lowers every token and reaches the native call unchanged.
    system().set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")


def test_user_flux_stays_a_plain_token():
    # Group 9: the external-flux "user" token has no native route; it resolves through the
    # external-brick catalog, so the scheme table keeps it as a plain string.
    assert _FLUX_SCHEMES["user"] == "user"
    assert not isinstance(_FLUX_SCHEMES["user"], routes.Route)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
