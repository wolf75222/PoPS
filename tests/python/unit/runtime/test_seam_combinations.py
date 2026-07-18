"""Every generated native transport/flux seam is executable.

``src/runtime/builders/seam_combinations.cmake`` is the declarative authority for the
generated System and AMR builder translation units.  The source-only architecture gate
checks that manifest against the component registry; this runtime gate complements it by
advancing every declared route through the final ``add_equation`` engine seam.  A missing
or miswired generated constructor therefore fails here instead of surviving as dead code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import HLL, HLLC, Roe, Rusanov
import pops.runtime._engine_descriptors as engine
from pops.runtime._system import AmrSystem, System


_COMBINATIONS = (
    ("exb", None),
    ("isothermal", "rusanov"),
    ("isothermal", "hll"),
    ("compressible", "rusanov"),
    ("compressible", "hll"),
    ("compressible", "hllc"),
    ("compressible", "roe"),
)
_FLUX_TYPES = {
    "rusanov": Rusanov,
    "hll": HLL,
    "hllc": HLLC,
    "roe": Roe,
}


def _model(transport: str) -> engine.Model:
    if transport == "exb":
        return engine.Model(
            state=engine.Scalar(),
            transport=engine.ExB(B0=1.0),
            source=engine.NoSource(),
            elliptic=engine.BackgroundDensity(alpha=1.0, n0=1.0),
        )
    if transport == "isothermal":
        return engine.Model(
            state=engine.FluidState("isothermal", cs2=0.5),
            transport=engine.IsothermalFlux(),
            source=engine.PotentialForce(charge=1.0),
            elliptic=engine.BackgroundDensity(alpha=1.0, n0=1.0),
        )
    if transport == "compressible":
        return engine.Model(
            state=engine.FluidState("compressible", gamma=1.4),
            transport=engine.CompressibleFlux(),
            source=engine.PotentialForce(charge=-1.0),
            elliptic=engine.BackgroundDensity(alpha=-1.0, n0=1.0),
        )
    raise AssertionError("unknown manifest transport %r" % transport)


def _spatial(transport: str, flux: str | None) -> engine.Spatial:
    if flux is None:
        return engine.Spatial(minmod=True)
    return engine.Spatial(
        limiter=Minmod(),
        flux=_FLUX_TYPES[flux](),
        primitive=transport == "compressible",
    )


def _seed_density(runtime: System | AmrSystem, name: str, n: int) -> None:
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    density = 1.0 + 0.1 * np.sin(2.0 * math.pi * xx) * np.sin(2.0 * math.pi * yy)
    runtime.set_density(name, density)


@pytest.mark.parametrize(("transport", "flux"), _COMBINATIONS)
def test_system_generated_seam_advances(transport: str, flux: str | None) -> None:
    n = 32
    runtime = System(n=n, L=1.0, periodic=True)
    runtime.add_equation(
        "block",
        _model(transport),
        spatial=_spatial(transport, flux),
    )
    _seed_density(runtime, "block", n)
    dt = runtime.step_cfl(0.4)
    assert math.isfinite(dt) and dt > 0.0


@pytest.mark.parametrize(("transport", "flux"), _COMBINATIONS)
def test_amr_generated_seam_advances(transport: str, flux: str | None) -> None:
    n = 32
    runtime = AmrSystem(n=n, regrid_every=0, periodic=True)
    runtime.add_equation(
        "block",
        _model(transport),
        spatial=_spatial(transport, flux),
    )
    _seed_density(runtime, "block", n)
    dt = runtime.step_cfl(0.4)
    assert math.isfinite(dt) and dt > 0.0
