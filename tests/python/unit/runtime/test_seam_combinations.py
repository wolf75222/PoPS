"""Every built-in route materializes and executes through the exact-ranked AMR package."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import HLL, HLLC, Roe, Rusanov
import pops.runtime._engine_descriptors as engine
from pops.runtime._system import AmrSystem
from tests.python.support.explicit_program import install_forward_euler_program


_COMBINATIONS = (
    ("exb", None),
    ("isothermal", "rusanov"),
    ("isothermal", "hll"),
    ("isothermal", "hllc"),
    ("isothermal", "roe"),
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
            transport=engine.ExB(),
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


def _seed_density(runtime: AmrSystem, name: str, n: int) -> None:
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    density = 1.0 + 0.1 * np.sin(2.0 * math.pi * xx) * np.sin(2.0 * math.pi * yy)
    runtime.set_density(name, density)


@pytest.mark.parametrize(("transport", "flux"), _COMBINATIONS)
def test_amr_prepared_package_route_advances(transport: str, flux: str | None) -> None:
    n = 32
    runtime = AmrSystem(n=n, regrid_every=0, periodicity=(True, True))
    runtime.add_equation(
        "block",
        _model(transport),
        spatial=_spatial(transport, flux),
    )
    _seed_density(runtime, "block", n)
    install_forward_euler_program(runtime)
    dt = runtime.step_cfl(0.4)
    assert math.isfinite(dt) and dt > 0.0
