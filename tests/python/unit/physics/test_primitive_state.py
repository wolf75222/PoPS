"""Native primitive/conservative conversion regression.

The final public lifecycle binds conservative initial values through ``Case``.  The native executor
still owns model-specific primitive conversion for diagnostics and internal initialisation, so this
test keeps that numerical oracle on the exact ``add_equation`` engine seam.  It does not restore the
retired ``System.block`` alias or any public compatibility method.
"""
import os
import shutil

import numpy as np
import pytest

import pops
import pops.runtime._engine_descriptors as engine
from pops.physics import Density, Energy, Model, Momentum, Pressure, Velocity
from pops.runtime._system import System
from tests.python.support.physics_roles import FRAME, X_AXIS, Y_AXIS
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


def _board_primitive_fixture(name):
    model = Model(name, frame=FRAME)
    state = model.state(
        "U",
        components=("rho", "rho_u", "rho_v", "E"),
        roles={
            "rho": Density(),
            "rho_u": Momentum(axis=X_AXIS),
            "rho_v": Momentum(axis=Y_AXIS),
            "E": Energy(),
        },
    )
    rho, rho_u, rho_v, energy = state
    velocity_x = model.primitive("u", rho_u / rho)
    velocity_y = model.primitive("v", rho_v / rho)
    pressure = model.primitive(
        "p", (1.4 - 1.0) * (
            energy - 0.5 * rho * (velocity_x * velocity_x + velocity_y * velocity_y)
        ),
    )
    components = (rho, velocity_x, velocity_y, pressure)
    inverse = (
        rho,
        rho * velocity_x,
        rho * velocity_y,
        pressure / (1.4 - 1.0)
        + 0.5 * rho * (velocity_x * velocity_x + velocity_y * velocity_y),
    )
    roles = {
        "rho": Density(),
        "u": Velocity(axis=X_AXIS),
        "v": Velocity(axis=Y_AXIS),
        "p": Pressure(),
    }
    return model, components, inverse, roles


def _primitive_contract_snapshot(model):
    hyp = model._dsl._m
    return (
        tuple(hyp.prim_state),
        None if hyp.prim_roles is None else tuple(hyp.prim_roles),
        None if hyp.cons_from is None else tuple(repr(value) for value in hyp.cons_from),
        model._primitive_state_authored,
    )


def test_board_primitive_state_declares_one_typed_atomic_coordinate_system():
    model, components, inverse, roles = _board_primitive_fixture("primitive_board_contract")
    model.primitive_state(*components, conservative=inverse, roles=roles)

    hyp = model._dsl._m
    assert hyp.prim_state == ["rho", "u", "v", "p"]
    assert hyp.prim_roles == ["Density", "VelocityX", "VelocityY", "Pressure"]
    assert tuple(repr(value) for value in hyp.cons_from) == tuple(
        repr(value) for value in inverse
    )
    assert model._primitive_state_authored is True


def test_board_primitive_state_rejects_foreign_arity_and_rolls_back_builder_failure(
    monkeypatch,
):
    model, components, inverse, roles = _board_primitive_fixture("primitive_board_atomic")
    foreign, foreign_components, _, _ = _board_primitive_fixture("primitive_board_foreign")
    del foreign
    before = _primitive_contract_snapshot(model)

    with pytest.raises(ValueError, match="not declared by this physics model"):
        model.primitive_state(
            components[0], foreign_components[1], components[2], components[3],
            conservative=inverse, roles=roles,
        )
    assert _primitive_contract_snapshot(model) == before

    with pytest.raises(ValueError, match="matching state"):
        model.primitive_state(*components[:-1], conservative=inverse, roles=roles)
    assert _primitive_contract_snapshot(model) == before

    foreign_inverse = (foreign_components[0], *inverse[1:])
    with pytest.raises(ValueError, match="not an owned selected primitive component"):
        model.primitive_state(*components, conservative=foreign_inverse, roles=roles)
    assert _primitive_contract_snapshot(model) == before

    def fail_after_inverse_mutation(expressions):
        model._dsl._m.cons_from = list(expressions)
        raise RuntimeError("injected primitive inverse failure")

    monkeypatch.setattr(model._dsl, "conservative_from", fail_after_inverse_mutation)
    with pytest.raises(RuntimeError, match="injected primitive inverse failure"):
        model.primitive_state(*components, conservative=inverse, roles=roles)
    assert _primitive_contract_snapshot(model) == before


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
    """The final artifact preserves primitive metadata and runs the bound conservative state."""
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not compiler or not os.path.isdir(INCLUDE):
        return

    from tests.python.unit.codegen.test_dsl_coupled import (
        GAMMA,
        build_euler,
        compile_euler_artifact,
    )

    rho, u, v, p = _fields()
    initial = np.ascontiguousarray(np.stack((
        rho,
        rho * u,
        rho * v,
        p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v),
    )))

    # compile_euler_artifact owns the explicit public
    # Case -> validate -> resolve -> compile transaction.  This test then uses
    # only the final bind/run/read surface; it neither detaches the component
    # into System nor reconstructs a second numerical plan around the package.
    artifact = compile_euler_artifact(
        build_euler("primitive_roundtrip_production"), cells=N, cxx=compiler,
    )
    component = artifact.blocks[0].model
    assert component.backend == "production"
    assert component.prim_names == ["rho", "u", "v", "p"]

    simulation = pops.bind(artifact, initial_state={"gas": initial.copy()})
    bound = np.asarray(simulation.state_global("gas"), dtype=np.float64).reshape(initial.shape)
    bound_rho = bound[0]
    bound_u = bound[1] / bound_rho
    bound_v = bound[2] / bound_rho
    bound_p = (GAMMA - 1.0) * (
        bound[3] - 0.5 * bound_rho * (bound_u * bound_u + bound_v * bound_v)
    )
    for actual, expected in zip(
        (bound_rho, bound_u, bound_v, bound_p), (rho, u, v, p), strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13)

    report = pops.run(simulation, t_end=5.0e-4, max_steps=5)
    state = np.asarray(simulation.state_global("gas"), dtype=np.float64).reshape(initial.shape)
    assert report.accepted_steps == simulation.macro_step() == 5
    assert np.isfinite(state).all() and state[0].min() > 0.0
    assert float(np.max(np.abs(state - initial))) > 1.0e-8
