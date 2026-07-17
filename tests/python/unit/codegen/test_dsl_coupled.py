"""Shared final Euler fixture for native-package integration tests.

The fixture deliberately enters compilation through the public
``Case -> validate -> resolve -> compile`` lifecycle.  Native-loader tests may
then detach the resulting :class:`CompiledModel` to exercise the low-level
runtime boundary, but no test authors a second compiler or calls the retired
``HyperbolicModel.compile`` shortcut.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pops
from pops.codegen import Production
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import ConstParam
from pops.physics import Density, Energy, Model, Momentum, Pressure, Velocity
from pops.time import FixedDt

from tests.python.support.physics_roles import FRAME, X_AXIS, Y_AXIS
from tests.python.support.requirements import repo_include


GAMMA = 1.4
INCLUDE = repo_include()
ROOT = Path(__file__).resolve().parents[4]


def build_euler(name: str = "euler") -> Model:
    """Return one typed 2-D Euler model with explicit HLLC capability."""
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
    gamma = model.value(model.param(ConstParam("gamma", GAMMA)))
    velocity_x = model.primitive("u", rho_u / rho)
    velocity_y = model.primitive("v", rho_v / rho)
    pressure = model.primitive(
        "p",
        (gamma - 1.0)
        * (energy - 0.5 * (rho_u * rho_u + rho_v * rho_v) / rho),
    )
    model.primitive_state(
        rho,
        velocity_x,
        velocity_y,
        pressure,
        conservative=(
            rho,
            rho * velocity_x,
            rho * velocity_y,
            pressure / (gamma - 1.0)
            + 0.5 * rho * (velocity_x * velocity_x + velocity_y * velocity_y),
        ),
        roles={
            "rho": Density(),
            "u": Velocity(axis=X_AXIS),
            "v": Velocity(axis=Y_AXIS),
            "p": Pressure(),
        },
    )
    sound_speed = model.scalar("c", sqrt(gamma * pressure / rho))
    flux = model.flux(
        "transport",
        frame=FRAME,
        state=state,
        components={
            X_AXIS: (
                rho * velocity_x,
                rho_u * velocity_x + pressure,
                rho_v * velocity_x,
                (energy + pressure) * velocity_x,
            ),
            Y_AXIS: (
                rho * velocity_y,
                rho_u * velocity_y,
                rho_v * velocity_y + pressure,
                (energy + pressure) * velocity_y,
            ),
        },
        waves={
            X_AXIS: (
                velocity_x - sound_speed,
                velocity_x,
                velocity_x,
                velocity_x + sound_speed,
            ),
            Y_AXIS: (
                velocity_y - sound_speed,
                velocity_y,
                velocity_y,
                velocity_y + sound_speed,
            ),
        },
    )
    # HLLC is a model capability, while the actual numerical choice remains in
    # DiscretizationPlan.  Pressure is the supported single-state formula hook;
    # sound speed is already part of the typed flux-wave contract above and the
    # role-derived HLLC provider owns its multi-state acoustic construction.
    model.riemann(riemann.HLLC(), pressure=pressure)
    model.rate("transport", equation=ddt(state) == -div(flux))
    return model


def compile_euler_artifact(
    model: Model,
    *,
    cells: int = 16,
    cxx: str | None = None,
) -> Any:
    """Compile ``model`` through the final lifecycle and return the simulation artifact."""
    state = model.states["U"]
    flux = model.fluxes["transport"]
    rate = model.operators["transport"]

    case = pops.Case("%s-native-package-case" % model.name)
    block = case.block("gas", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.MUSCL(limiters.Minmod()),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(1.0e-4))
    case.program(program)
    layout = Uniform(
        CartesianGrid(
            frame=model.frame,
            cells=(cells, cells),
            periodic=PeriodicAxes(model.frame.axes),
        )
    )
    options: dict[str, Any] = {"include": str(ROOT / "include")}
    if cxx is not None:
        options["cxx"] = cxx
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options=options,
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    return artifact


def compile_euler_component(
    model: Model,
    *,
    cells: int = 16,
    cxx: str | None = None,
) -> Any:
    """Return the one detached component produced by :func:`compile_euler_artifact`."""
    artifact = compile_euler_artifact(model, cells=cells, cxx=cxx)
    if len(artifact.blocks) != 1:
        raise AssertionError("Euler fixture must compile exactly one block component")
    return artifact.blocks[0].model


def test_shared_euler_fixture_is_operator_first() -> None:
    model = build_euler("shared-euler-contract")
    assert tuple(model.states) == ("U",)
    assert tuple(model.fluxes) == ("transport",)
    assert "transport" in model.operators
    registered_rate = model.module.operator_registry().get("transport")
    assert registered_rate.kind == "local_rate"
