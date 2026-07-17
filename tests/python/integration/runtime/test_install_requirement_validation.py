"""Spec 2 (ADC-446, criterion 24): install-time operator-requirement validation.

A compiled problem.so carries, per operator, the aux fields its body reads (the GeneratedModule
descriptor). System.install_program reads that descriptor and rejects, BEFORE installing the program,
a simulation that did not provide a required field -- here B_z, normally supplied by
set_magnetic_field -- with a spec-style message ("operator 'lorentz' requires aux field 'B_z', but
simulation did not provide it") instead of a cryptic failure mid-step. The negative and positive
cases both need a compiler + a visible Kokkos (POPS_KOKKOS_ROOT) to build the .so. The exact native
preflight is an explicit optional local skip and a fail-closed requirement in native CI. Any later
compile failure propagates as a real regression. cf. docs/sphinx/reference/operator-modules.md.
"""
import sys

from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)

try:
    import numpy as np

    import pops.runtime._engine_descriptors as engine
    import pops.lib.time as libtime
    from pops.codegen._compile_drivers import compile_problem
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.math import ddt, div, sqrt
    from pops.physics import Model
    from pops.problem import Case
    from pops.runtime._system import System  # ADC-545 advanced runtime seam
    from tests.python.integration._final_field_program import compile_block_model
except Exception as exc:  # noqa: BLE001
    require_native_or_skip(
        "test_install_requirement_validation imports unavailable: %s" % exc
    )

N = 16


def lorentz_model(name="adc446_model"):
    """An isothermal fluid whose Lorentz linear source reads the aux field B_z (a hard requirement)."""
    frame = Rectangle(
        "%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    m = Model(name, frame=frame)
    state = m.state("U", components=("rho", "mx", "my"))
    rho, mx, my = state
    cs = sqrt(0.5)
    flux = m.flux(
        "transport",
        frame=frame,
        state=state,
        components={
            x_axis: (mx, mx * mx / rho + 0.5 * rho, mx * my / rho),
            y_axis: (my, mx * my / rho, my * my / rho + 0.5 * rho),
        },
        waves={
            x_axis: (mx / rho - cs, mx / rho, mx / rho + cs),
            y_axis: (my / rho - cs, my / rho, my / rho + cs),
        },
    )
    bz = m.aux("B_z")
    m.operator(
        "lorentz",
        returns=m.local_linear_operator(
            "lorentz",
            on=state,
            matrix=((0.0, 0.0, 0.0), (0.0, 0.0, bz), (0.0, -bz, 0.0)),
        ),
    )
    m.rate("explicit_rhs", equation=ddt(state) == -div(flux))
    return m


def lie_program(model, name="adc446_prog"):
    case = Case("%s-case" % name)
    state = case.block("plasma", model)[model.states["U"]]
    return libtime.ForwardEuler(
        state,
        rate=model.operators["explicit_rhs"],
    )


def make_sim(block_model, with_bz):
    sim = System(n=N, L=1.0, periodic=True)
    sim.add_equation("plasma", compile_block_model(block_model, target="system"),
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")
    if with_bz:
        sim.set_magnetic_field(3.0 * np.ones(N * N))
    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    sim.set_state("plasma", np.stack([rho, 0.4 * rho, -0.2 * rho]))
    return sim


def main():
    missing = missing_native_compile_requirement(repo_include(), default_cxx())
    if missing:
        require_native_or_skip("test_install_requirement_validation: %s" % missing)
    if not hasattr(System(n=8, L=1.0, periodic=True), "install_program"):
        require_native_or_skip(
            "test_install_requirement_validation requires System.install_program"
        )
    m = lorentz_model()
    compiled = compile_problem(model=m, time=lie_program(m))

    # (1) Negative: a simulation WITHOUT set_magnetic_field must be rejected at install with the
    # spec-style message naming the operator and the missing aux field.
    sim_missing = make_sim(m, with_bz=False)
    try:
        sim_missing.install_program(compiled.so_path)
        raise AssertionError("install accepted a simulation missing B_z; expected a RuntimeError "
                             "naming operator 'lorentz' and aux 'B_z'")
    except RuntimeError as exc:
        msg = str(exc)
        ok = "lorentz" in msg and "B_z" in msg and "did not provide" in msg
        assert ok, ("install rejection message must name operator 'lorentz', aux 'B_z' and "
                    "'did not provide'; got: %s" % msg)
        print("OK  install rejects a missing required aux: %s" % msg)

    # (2) Positive: providing B_z (set_magnetic_field) lets the same program install cleanly.
    sim_ok = make_sim(m, with_bz=True)
    sim_ok.install_program(compiled.so_path)
    print("OK  install accepts the simulation once B_z is provided")
    return 0


if __name__ == "__main__":
    sys.exit(main())
