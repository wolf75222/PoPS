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
    transport = str(spec.transport)
    density = None
    momentum_x = momentum_y = energy = None
    if transport == "exb":
        (density,) = wrapper.conservative_vars("n", roles=(Density(),))
        grad_x = wrapper.aux("grad_x")
        grad_y = wrapper.aux("grad_y")
        # Cartesian E×B with B=(0,0,1): v = (-d_y phi, d_x phi) / |B|^2.
        # Eigenvalues stay density-scaled zeros so AMR BoundFluxProviders used by
        # max_wave_speed do not require aux slots the composite pack does not forward.
        wrapper.flux(x=[density * (-grad_y)], y=[density * grad_x])
        wrapper.eigenvalues(x=[0.0 * density], y=[0.0 * density])
        wrapper.primitive_vars(density)
        wrapper.conservative_from([density])
    elif transport == "isothermal":
        density, momentum_x, momentum_y = wrapper.conservative_vars(
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
        u = wrapper.primitive("u", momentum_x / denom)
        v = wrapper.primitive("v", momentum_y / denom)
        pressure = wrapper.primitive("p", cs2 * density)
        wrapper.flux(
            x=[momentum_x, momentum_x * u + pressure, momentum_x * v],
            y=[momentum_y, momentum_y * u, momentum_y * v + pressure],
        )
        sound = sqrt(pressure / density)
        wrapper.eigenvalues(x=[u - sound, u, u + sound], y=[v - sound, v, v + sound])
        wrapper.primitive_vars(
            density,
            u,
            v,
            roles=(Density(), Velocity(axis=x_axis), Velocity(axis=y_axis)),
        )
        wrapper.conservative_from([density, density * u, density * v])
    elif transport == "compressible":
        density, momentum_x, momentum_y, energy = wrapper.conservative_vars(
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
        wrapper.gamma(gamma)
        u = wrapper.primitive("u", momentum_x / density)
        v = wrapper.primitive("v", momentum_y / density)
        pressure = wrapper.primitive(
            "p", (gamma - 1.0) * (energy - 0.5 * density * (u * u + v * v))
        )
        enthalpy = (energy + pressure) / density
        sound = sqrt(gamma * pressure / density)
        wrapper.flux(
            x=[momentum_x, momentum_x * u + pressure, momentum_x * v, density * enthalpy * u],
            y=[momentum_y, momentum_y * u, momentum_y * v + pressure, density * enthalpy * v],
        )
        wrapper.eigenvalues(x=[u - sound, u, u + sound], y=[v - sound, v, v + sound])
        wrapper.primitive_vars(
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
        wrapper.conservative_from(
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

    elliptic = str(getattr(spec, "elliptic", "") or "")
    if elliptic == "charge":
        wrapper.elliptic_rhs(float(spec.q) * density)
    elif elliptic == "background":
        wrapper.elliptic_rhs(float(spec.alpha) * (density - float(spec.n0)))
    elif elliptic == "gravity":
        wrapper.elliptic_rhs(
            float(spec.sign) * float(spec.four_pi_G) * (density - float(spec.rho0))
        )
    elif elliptic not in ("", "none"):
        raise ValueError(
            "ModelSpec elliptic %r cannot be compiled; expected charge, background, or gravity"
            % elliptic
        )

    # Authored source(aux) emits provider_value<0> but the AMR generated source
    # path currently materializes ProviderValues<0>. PotentialForce stays a
    # native brick concern; the hyperbolic/elliptic package still compiles.

    if transport in ("isothermal", "compressible"):
        wrapper.enable_hllc()
        wrapper.enable_roe()
    wrapper._invalidate_authoring_views()

    compiled = wrapper.compile(
        backend="production",
        target=target,
        name=name,
        consumer_owner_qid="pops.runtime.modelspec.%s" % name,
    )
    if type(compiled) is not CompiledModel:
        raise TypeError("ModelSpec compilation did not produce a CompiledModel")
    return compiled
