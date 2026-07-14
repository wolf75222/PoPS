"""Native primitive/conservative conversion regression.

The final public lifecycle binds conservative initial values through ``Case``.  The native executor
still owns model-specific primitive conversion for diagnostics and internal initialisation, so this
test keeps that numerical oracle on the exact ``add_equation`` engine seam.  It does not restore the
retired ``System.block`` alias or any public compatibility method.
"""
import os
import shutil
import tempfile

import numpy as np

from pops.codegen import Production
import pops.runtime._engine_descriptors as engine
from pops.runtime._system import System
from tests.python.support.requirements import repo_include


N, L = 24, 1.0
INCLUDE = repo_include()


def _fields(n=N):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    rho = 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
    u = 0.2 * np.sin(2 * np.pi * X)
    v = -0.1 * np.cos(2 * np.pi * Y)
    p = 0.5 + 0.1 * X
    return rho, u, v, p


def _runtime(name, model):
    runtime = System(n=N, L=L, periodic=True)
    runtime.add_equation(
        name,
        model,
        spatial=engine.Spatial(minmod=True),
        time=engine.Explicit(),
    )
    return runtime


def _roundtrip_error(runtime, name, **primitives):
    runtime.set_primitive_state(name, **primitives)
    result = runtime.get_primitive_state(name)
    return max(
        float(np.max(np.abs(result[key] - np.asarray(value))))
        for key, value in primitives.items()
    )


def test_compressible_primitive_roundtrip_and_step():
    rho, u, v, p = _fields()
    runtime = _runtime(
        "gas",
        engine.Model(
            state=engine.FluidState("compressible", gamma=1.4),
            transport=engine.CompressibleFlux(),
            source=engine.NoSource(),
            elliptic=engine.ChargeDensity(charge=1.0),
        ),
    )
    assert runtime.variable_names("gas", "primitive") == ["rho", "u", "v", "p"]
    assert _roundtrip_error(runtime, "gas", rho=rho, u=u, v=v, p=p) < 1e-13

    state = np.asarray(runtime.get_state("gas")).reshape(4, N, N)
    expected_energy = p / (1.4 - 1.0) + 0.5 * rho * (u * u + v * v)
    np.testing.assert_allclose(state[1], rho * u, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(state[2], rho * v, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(state[3], expected_energy, rtol=0.0, atol=1e-13)

    runtime.set_poisson()
    for _ in range(5):
        runtime.step_cfl(0.4)
    advanced = np.asarray(runtime.get_state("gas")).reshape(4, N, N)
    assert np.isfinite(advanced).all() and advanced[0].min() > 0.0


def test_isothermal_primitive_roundtrip_and_step():
    rho, u, v, _ = _fields()
    runtime = _runtime(
        "gas",
        engine.Model(
            state=engine.FluidState("isothermal", cs2=0.5),
            transport=engine.IsothermalFlux(),
            source=engine.NoSource(),
            elliptic=engine.ChargeDensity(charge=1.0),
        ),
    )
    assert runtime.variable_names("gas", "primitive") == ["rho", "u", "v"]
    assert _roundtrip_error(runtime, "gas", rho=rho, u=u, v=v) < 1e-13
    state = np.asarray(runtime.get_state("gas")).reshape(3, N, N)
    np.testing.assert_allclose(state[1], rho * u, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(state[2], rho * v, rtol=0.0, atol=1e-13)

    runtime.set_poisson()
    for _ in range(5):
        runtime.step_cfl(0.4)
    advanced = np.asarray(runtime.get_state("gas")).reshape(3, N, N)
    assert np.isfinite(advanced).all() and advanced[0].min() > 0.0


def test_scalar_conversion_is_identity():
    rho, _, _, _ = _fields()
    runtime = _runtime(
        "tracer",
        engine.Model(
            state=engine.Scalar(),
            transport=engine.ExB(B0=1.0),
            source=engine.NoSource(),
            elliptic=engine.ChargeDensity(charge=1.0),
        ),
    )
    primitive, = runtime.variable_names("tracer", "primitive")
    assert _roundtrip_error(runtime, "tracer", **{primitive: rho}) < 1e-15


def test_set_density_preserves_rest_state_and_energy_default():
    rho, _, _, _ = _fields()
    runtime = _runtime(
        "gas",
        engine.Model(
            state=engine.FluidState("compressible", gamma=1.4),
            transport=engine.CompressibleFlux(),
            source=engine.NoSource(),
            elliptic=engine.ChargeDensity(charge=1.0),
        ),
    )
    runtime.set_density("gas", rho)
    state = np.asarray(runtime.get_state("gas")).reshape(4, N, N)
    np.testing.assert_array_equal(state[0], rho)
    assert np.count_nonzero(state[1]) == 0 and np.count_nonzero(state[2]) == 0
    np.testing.assert_allclose(state[3], rho / (1.4 - 1.0), rtol=0.0, atol=1e-13)


def test_primitive_input_errors_name_missing_and_extra_components():
    rho, u, v, p = _fields()
    runtime = _runtime(
        "gas",
        engine.Model(
            state=engine.FluidState("compressible", gamma=1.4),
            transport=engine.CompressibleFlux(),
            source=engine.NoSource(),
            elliptic=engine.ChargeDensity(charge=1.0),
        ),
    )
    try:
        runtime.set_primitive_state("gas", rho=rho, u=u, v=v, p=p, bogus=p)
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("unknown primitive name must be rejected")

    try:
        runtime.set_primitive_state("gas", rho=rho, u=u)
    except ValueError as exc:
        assert "missing primitive(s)" in str(exc)
    else:
        raise AssertionError("missing primitive values must be rejected")


def test_production_model_primitive_roundtrip_and_step():
    """A production package forwards its authored conservative/primitive conversion."""
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not compiler or not os.path.isdir(INCLUDE):
        return

    from tests.python.unit.codegen.test_dsl_coupled import build_euler_poisson

    rho, u, v, p = _fields()
    directory = tempfile.mkdtemp()
    try:
        compiled = build_euler_poisson().compile(
            os.path.join(directory, "euler_poisson_native.so"),
            INCLUDE,
            backend=Production(),
        )
        runtime = System(n=N, L=L, periodic=True)
        runtime.add_equation(
            "gas",
            compiled,
            spatial=engine.Spatial(minmod=True),
            time=engine.Explicit(),
        )
        assert compiled.backend == "production"
        assert runtime.variable_names("gas", "primitive") == ["rho", "u", "v", "p"]
        assert _roundtrip_error(runtime, "gas", rho=rho, u=u, v=v, p=p) < 1e-13

        runtime.set_poisson(rhs="charge_density")
        for _ in range(5):
            runtime.step_cfl(0.4)
        state = np.asarray(runtime.get_state("gas")).reshape(4, N, N)
        assert np.isfinite(state).all() and state[0].min() > 0.0
    finally:
        shutil.rmtree(directory, ignore_errors=True)
