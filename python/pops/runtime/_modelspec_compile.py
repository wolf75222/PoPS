"""Compile a private native ModelSpec into one production Prepared*Block package.

``System.add_block(ModelSpec)`` / ``AmrSystem.add_block(ModelSpec)`` stay retired. The remaining
Python ``add_equation(ModelSpec)`` seam authors the equivalent typed hyperbolic model, compiles it
for the loaded native rank, and installs the resulting ``CompiledModel``.
"""

from __future__ import annotations

from typing import Any


def compile_modelspec_package(
    spec: Any,
    *,
    name: str,
    target: str,
) -> Any:
    """Return a detached production ``CompiledModel`` for one ModelSpec brick composition."""
    from pops._bootstrap import ModelSpec
    from pops.codegen.loader import CompiledModel
    from pops.codegen.toolchain import loader_native_dimension
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.math import maximum, sqrt
    from pops.physics import Density, Energy, Momentum, Pressure, Velocity
    from pops.physics._facade import Model

    if not isinstance(spec, ModelSpec):
        raise TypeError("compile_modelspec_package requires a private ModelSpec")
    if target not in ("system", "amr_system"):
        raise ValueError("compile_modelspec_package target must be system or amr_system")
    if loader_native_dimension() != 2:
        raise ValueError(
            "ModelSpec compilation currently prepares the loaded 2-D native specialization"
        )

    frame = Rectangle("modelspec-domain", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    wrapper = Model("%s_modelspec" % name)
    model = wrapper._m
    transport = str(spec.transport)
    if transport == "exb":
        (density,) = model.conservative_vars("n", roles=(Density(),))
        model.set_flux(x=[0.0 * density], y=[0.0 * density])
        model.set_eigenvalues(x=[0.0 * density], y=[0.0 * density])
        model.set_primitive_state(density)
        model.set_conservative_from([density])
    elif transport == "isothermal":
        density, momentum_x, momentum_y = model.conservative_vars(
            "rho",
            "mx",
            "my",
            roles=(
                Density(),
                Momentum(axis=x_axis),
                Momentum(axis=y_axis),
            ),
        )
        cs2 = float(spec.cs2)
        floor = float(spec.vacuum_floor)
        denom = density if floor == 0.0 else maximum(density, floor)
        u = model.primitive("u", momentum_x / denom)
        v = model.primitive("v", momentum_y / denom)
        pressure = model.primitive("p", cs2 * density)
        model.set_flux(
            x=[momentum_x, momentum_x * u + pressure, momentum_x * v],
            y=[momentum_y, momentum_y * u, momentum_y * v + pressure],
        )
        sound = sqrt(pressure / density)
        model.set_eigenvalues(x=[u - sound, u, u + sound], y=[v - sound, v, v + sound])
        model.set_primitive_state(
            density,
            u,
            v,
            roles=(Density(), Velocity(axis=x_axis), Velocity(axis=y_axis)),
        )
        model.set_conservative_from([density, density * u, density * v])
    elif transport == "compressible":
        density, momentum_x, momentum_y, energy = model.conservative_vars(
            "rho",
            "mx",
            "my",
            "E",
            roles=(
                Density(),
                Momentum(axis=x_axis),
                Momentum(axis=y_axis),
                Energy(),
            ),
        )
        gamma = float(spec.gamma)
        model.set_gamma(gamma)
        u = model.primitive("u", momentum_x / density)
        v = model.primitive("v", momentum_y / density)
        pressure = model.primitive(
            "p", (gamma - 1.0) * (energy - 0.5 * density * (u * u + v * v))
        )
        enthalpy = (energy + pressure) / density
        sound = sqrt(gamma * pressure / density)
        model.set_flux(
            x=[momentum_x, momentum_x * u + pressure, momentum_x * v, density * enthalpy * u],
            y=[momentum_y, momentum_y * u, momentum_y * v + pressure, density * enthalpy * v],
        )
        model.set_eigenvalues(x=[u - sound, u, u + sound], y=[v - sound, v, v + sound])
        model.set_primitive_state(
            density,
            u,
            v,
            pressure,
            roles=(
                Density(),
                Velocity(axis=x_axis),
                Velocity(axis=y_axis),
                Pressure(),
            ),
        )
        model.set_conservative_from(
            [
                density,
                density * u,
                density * v,
                pressure / (gamma - 1.0) + 0.5 * density * (u * u + v * v),
            ]
        )
    else:
        raise ValueError(
            "ModelSpec transport %r cannot be compiled; expected exb, isothermal, or compressible"
            % transport
        )

    if transport in ("isothermal", "compressible"):
        wrapper.enable_hllc()
        wrapper.enable_roe()

    compiled = wrapper.compile(
        backend="production",
        target=target,
        name=name,
        consumer_owner_qid="pops.runtime.modelspec.%s" % name,
    )
    if type(compiled) is not CompiledModel:
        raise TypeError("ModelSpec compilation did not produce a CompiledModel")
    return compiled
