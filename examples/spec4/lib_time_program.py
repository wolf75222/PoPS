#!/usr/bin/env python3
"""Spec 4 (31): the same scheme from the library instead of by hand.

``examples/spec4/manual_time_program.py`` spells a predictor-corrector / backward-Euler
step out of kernel primitives. Here the SAME scheme comes from the time-scheme library:

    from pops.lib.time import predictor_corrector_local_linear

The library scheme is a MACRO: given a bound model and typed operator handles, it emits the
same kernel primitives into the Program. This script builds the model, applies the macro,
prints the emitted IR, and (when a toolchain is present) compiles the Program to a
``problem.so``.

This example is intentionally fail-loud: if the compiler, Kokkos, or lowering path is missing,
``pops.compile_problem`` raises instead of hiding the failure.

Run::

    python3 examples/spec4/lib_time_program.py
"""
import sys

from pops.lib.time import predictor_corrector_local_linear
from pops.math import ddt, div, grad, laplacian, sqrt
from pops.physics import Model
from pops.solvers.elliptic import GeometricMG
from pops.time import Program


def build_model():
    """2D isothermal Euler + Poisson + Lorentz, identical physics to the manual example."""
    m = Model("euler_poisson_lorentz")
    state = m.state(
        "U",
        components=["rho", "mx", "my"],
        roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"},
    )
    rho, mx, my = state
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    cs2 = m.param("cs2", 1.0)
    pressure = m.scalar("p", cs2 * rho)
    sound = m.scalar("c", sqrt(cs2))
    flux = m.flux(
        "F",
        on=state,
        x=[mx, mx * u + pressure, mx * v],
        y=[my, my * u, my * v + pressure],
        waves={"x": [u - sound, u, u + sound], "y": [v - sound, v, v + sound]},
    )
    phi = m.field("phi")
    m.solve_field(
        "fields_from_state",
        equation=(-laplacian(phi) == rho),
        outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
        solver=GeometricMG(),
    )
    e_field = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
    electric = m.source("electric", on=state, value=[0.0 * rho, rho * e_field.x, rho * e_field.y])
    bz = m.aux("B_z")
    lorentz = m.local_linear_operator(
        "lorentz",
        on=state,
        matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]],
    )
    m.rate("explicit_rate", ddt(state) == -div(flux) + electric)
    m.operator("implicit_operator", returns=lorentz, inputs=["fields"])
    m.check()
    return m


def library_program(module):
    """Apply the library predictor-corrector macro against the bound operators."""
    program = Program("lib_predictor_corrector")
    program.bind_operators(module)
    ops = module.operator_registry()
    predictor_corrector_local_linear(
        program,
        "plasma",
        fields_operator=ops.get("fields_from_state"),
        explicit_rate_operator=ops.get("explicit_rate"),
        implicit_operator=ops.get("implicit_operator"),
    )
    return program


def main():
    model = build_model()
    module = model.to_module()
    program = library_program(module)

    print("model:", module.name)
    print("ops emitted by the macro:", sorted({v.op for v in program._values}))
    print("program nodes:", len(program._values))
    print("program commits:", {b: s.op for b, s in program.commits().items()})

    import pops

    compiled = pops.compile_problem(model=module, program=program)
    print("problem.so:", compiled.so_path)
    print("OK: library Spec 4 time scheme compiled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
