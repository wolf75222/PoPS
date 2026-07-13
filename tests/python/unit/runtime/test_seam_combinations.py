#!/usr/bin/env python3
"""ADC-593 runtime sibling of the seam-manifest architecture gate.

The per-route block-build seam TUs (system/{isothermal,compressible}, amr/block, amr/compiled) are now
GENERATED from python/bindings/seam_combinations.cmake instead of hand-written one file per (transport,
flux). This test proves the generation is not just source-equivalent but FUNCTIONAL: it drives a native
System (and AmrSystem) add_block for EVERY (transport, flux) combination the manifest declares, then
asserts a CFL step advances by a finite, positive dt. If a generated seam were mis-wired (wrong ctor,
wrong flux maker), the block would fail to build or the step would not advance.

It never fakes the engine: it builds a real System / AmrSystem via the public brick API and
skips cleanly if _pops (or numpy) is unavailable. Runs under pytest AND as a script (has __main__).
"""
import math
import sys

try:
    import numpy as np
    import pops
    from pops.numerics.riemann import HLL, HLLC, Roe, Rusanov
    from pops.numerics.reconstruction.limiters import Minmod
    from pops.runtime.system import AmrSystem, System  # ADC-545 advanced runtime seam
except Exception as exc:  # noqa: BLE001
    print("skip test_seam_combinations (pops unavailable: %s)" % exc)
    sys.exit(0)


# The manifest combinations, expressed in the public brick vocabulary. Kept in sync with
# python/bindings/seam_combinations.cmake by tests/python/architecture/test_pybind_seam_manifest.py (that gate
# locks the manifest to the catalog / registry; this list is the runtime projection of the same rows).
# (transport, flux) with flux=None for the transport-only (whole make_block dispatcher) seams.
_SYSTEM_COMBOS = [
    ("exb", None),
    ("isothermal", "rusanov"),
    ("isothermal", "hll"),
    ("compressible", "rusanov"),
    ("compressible", "hll"),
    ("compressible", "hllc"),
    ("compressible", "roe"),
]
# The AMR seam covers the same transports; the compressible flux leaves are the amr_block /
# amr_compiled per-flux TUs, exb / isothermal are the transport-only leaves.
_AMR_COMBOS = list(_SYSTEM_COMBOS)

_FLUX = {"rusanov": Rusanov, "hll": HLL, "hllc": HLLC, "roe": Roe}


def _model(transport):
    """A native pops.Model for @p transport (composed bricks, no DSL compile)."""
    if transport == "exb":
        return pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                          source=pops.NoSource(),
                          elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))
    if transport == "isothermal":
        return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                          transport=pops.IsothermalFlux(),
                          source=pops.PotentialForce(charge=1.0),
                          elliptic=pops.ChargeDensity(charge=1.0))
    if transport == "compressible":
        return pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                          transport=pops.CompressibleFlux(),
                          source=pops.PotentialForce(charge=-1.0),
                          elliptic=pops.ChargeDensity(charge=-1.0))
    raise AssertionError("unknown transport %r" % transport)


def _spatial(transport, flux):
    if flux is None:
        return pops.Spatial(minmod=True)
    # compressible fluxes reconstruct in primitive variables (matches test_bindings euler path).
    primitive = (transport == "compressible")
    return pops.Spatial(limiter=Minmod(), flux=_FLUX[flux](), primitive=primitive)


def _seed_density(sim, name, n):
    """A smooth positive density bump so the CFL wave speed is finite and the step advances."""
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.1 * np.sin(2 * math.pi * xx) * np.sin(2 * math.pi * yy)
    sim.set_density(name, rho)


from tests.python.support.assertions import _check


def test_system_seam_combinations():
    n = 32
    for transport, flux in _SYSTEM_COMBOS:
        label = "System %s/%s" % (transport, flux or "-")
        sim = System(n=n, L=1.0, periodic=True)
        sim.block("blk", model=_model(transport), spatial=_spatial(transport, flux))
        _seed_density(sim, "blk", n)
        dt = sim.step_cfl(0.4)
        _check(math.isfinite(dt) and dt > 0.0, "%s: step_cfl returned dt=%r" % (label, dt))
        print("  [OK ] %s advanced dt=%.3e" % (label, dt))


def test_amr_seam_combinations():
    nb = 32
    for transport, flux in _AMR_COMBOS:
        label = "AmrSystem %s/%s" % (transport, flux or "-")
        amr = AmrSystem(n=nb, regrid_every=0, periodic=True)
        amr.block("blk", model=_model(transport), spatial=_spatial(transport, flux))
        # AmrSystem seeds its coarse density through the same brick facade as System.
        _seed_density(amr, "blk", nb)
        dt = amr.step_cfl(0.4)
        _check(math.isfinite(dt) and dt > 0.0, "%s: step_cfl returned dt=%r" % (label, dt))
        print("  [OK ] %s advanced dt=%.3e" % (label, dt))


if __name__ == "__main__":
    test_system_seam_combinations()
    test_amr_seam_combinations()
    print("OK test_seam_combinations")
