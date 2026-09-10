"""Supported scalar and fluid routes execute through the exact-ranked AMR package."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import HLL, HLLC, Roe, Rusanov
import pops.runtime._engine_descriptors as engine
from pops.runtime._system import AmrSystem, AmrSystemConfig
from tests.python.support.explicit_program import install_forward_euler_program


_COMBINATIONS = (
    ("scalar_advection", None),
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


def _amr_config(n: int, *, regrid_every: int) -> AmrSystemConfig:
    config = AmrSystemConfig()
    config.shape = (n, n)
    config.lower = (0.0, 0.0)
    config.upper = (1.0, 1.0)
    config.periodicity = (True, True)
    config.boxes = (((0, 0), (n, n)),)
    config.regrid_every = regrid_every
    return config


def _model(transport: str):
    if transport == "scalar_advection":
        from pops.physics import Density
        from pops.physics._facade import Model

        model = Model("seam-scalar-advection")
        (rho,) = model.conservative_vars("n", roles=(Density(),))
        model.flux(x=[0.3 * rho], y=[0.2 * rho])
        model.eigenvalues(x=[0.3 + 0.0 * rho], y=[0.2 + 0.0 * rho])
        model.primitive_vars(rho)
        model.conservative_from([rho])
        model.elliptic_rhs(rho - 1.0)
        return model.compile(
            backend="production", target="amr_system", name="seam_scalar_advection",
            consumer_owner_qid="tests.seam-combinations.block",
        )
    if transport == "isothermal":
        return engine.Model(
            state=engine.FluidState("isothermal", cs2=0.5),
            transport=engine.IsothermalFlux(),
            source=engine.NoSource(),
            elliptic=engine.BackgroundDensity(alpha=1.0, n0=1.0),
        )
    if transport == "compressible":
        return engine.Model(
            state=engine.FluidState("compressible", gamma=1.4),
            transport=engine.CompressibleFlux(),
            source=engine.NoSource(),
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


def _seed_density(runtime: AmrSystem, name: str, n: int, transport: str) -> np.ndarray:
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    density = 1.0 + 0.1 * np.sin(2.0 * math.pi * xx) * np.sin(2.0 * math.pi * yy)
    if transport == "scalar_advection":
        runtime.set_density(name, density)
    else:
        components = [density, 0.3 * density, 0.2 * density]
        if transport == "compressible":
            pressure = 0.5 * density
            components.append(pressure / (1.4 - 1.0) + 0.5 * density * (0.3 ** 2 + 0.2 ** 2))
        runtime.set_conservative_state(name, np.stack(components))
    return density


@pytest.mark.parametrize(("transport", "flux"), _COMBINATIONS)
def test_amr_prepared_package_route_advances(transport: str, flux: str | None) -> None:
    n = 32
    runtime = AmrSystem(_amr_config(n, regrid_every=0))
    runtime.set_temporal_relations([2], [1], ["integral_only"])
    runtime.set_poisson(bc=engine.Periodic())
    runtime.add_equation(
        "block",
        _model(transport),
        spatial=_spatial(transport, flux),
    )
    initial_density = _seed_density(runtime, "block", n, transport)
    install_forward_euler_program(runtime)
    runtime.mark_bound()
    dt = runtime.step_cfl(0.4)
    assert math.isfinite(dt) and dt > 0.0
    density = np.asarray(runtime.density("block")).reshape(n, n)
    assert np.all(np.isfinite(density))
    assert np.max(np.abs(density - initial_density)) > 1.0e-6


def test_legacy_exb_requires_explicit_field_output_authority() -> None:
    runtime = AmrSystem(_amr_config(32, regrid_every=0))
    legacy = engine.Model(
        state=engine.Scalar(), transport=engine.ExB(), source=engine.NoSource(),
        elliptic=engine.BackgroundDensity(alpha=1.0, n0=1.0),
    )
    with pytest.raises(ValueError, match="ExB compilation requires an explicit field-output provider plan"):
        runtime.add_equation("block", legacy, spatial=engine.Spatial(minmod=True))
