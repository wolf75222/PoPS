"""Spec 2 (ADC-446, criterion 24): selected operator-requirement validation.

A compiled problem.so carries an exact provider plan for each selected operator. The assembling
install_program boundary permits inputs to be staged later; the first dependent evaluation must
refuse an unpublished B_z input before changing state or time. The diagnostic identifies its exact
physical provider, and the positive case stages that same ComponentKey. Merely declaring an unused
Lorentz operator must not require its input. The negative and positive
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
    from pops.codegen.component_provider_packs import resolve_component_provider_packs
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

    def transport(program, current, fraction, *, at):
        rate = model.operators["explicit_rhs"](current)
        return program.value("transported", current + fraction * program.dt * rate, at=at)

    def lorentz(program, current, fraction, *, at):
        rate = program.apply(model.operators["lorentz"], state=current)
        return program.value("rotated", current + fraction * program.dt * rate, at=at)

    # Only an actual selected read requires B_z. Merely declaring lorentz beside a transport-only
    # ForwardEuler program must not impose an unused provider requirement.
    return libtime.Lie(state, first=transport, second=lorentz)


def make_sim(block_model, with_bz):
    sim = System(n=N, L=1.0, periodicity=(True, True))
    sim.add_equation("plasma", compile_block_model(block_model, target="system"),
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method="euler"))
    sim.set_poisson("charge_density", "cartesian_cg")
    if with_bz:
        (magnetic_key,) = resolve_component_provider_packs(block_model.module).auxiliary
        sim.stage_auxiliary_input(magnetic_key, 3.0 * np.ones(N * N))
    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    sim.set_state("plasma", np.stack([rho, 0.4 * rho, -0.2 * rho]))
    return sim


def main():
    missing = missing_native_compile_requirement(repo_include(), default_cxx())
    if missing:
        require_native_or_skip("test_install_requirement_validation: %s" % missing)
    if not hasattr(System(n=8, L=1.0, periodicity=(True, True)), "install_program"):
        require_native_or_skip(
            "test_install_requirement_validation requires System.install_program"
        )
    m = lorentz_model()
    program = lie_program(m)
    assert sum(value.op == "apply" and value.attrs.get("linear_source") == "lorentz"
               for value in program._values) == 1
    compiled = compile_problem(model=m, time=program)
    (magnetic_key,) = resolve_component_provider_packs(m.module).auxiliary
    dt = 0.001

    # (1) Composition is still open at install; the selected Lorentz read is required to refuse
    # missing input at preparation, before any attempted state/time/step update can survive.
    sim_missing = make_sim(m, with_bz=False)
    before = np.array(sim_missing.get_state("plasma"))
    sim_missing.install_program(compiled.so_path)
    try:
        sim_missing.step(dt)
        raise AssertionError("selected Lorentz evaluation accepted missing B_z input")
    except RuntimeError as exc:
        msg = str(exc)
        assert "Program auxiliary prerequisite has never been published" in msg, msg
        assert magnetic_key.owner_qid in msg and "/fields/B_z" in msg, msg
        print("OK  selected Lorentz evaluation rejects its unpublished exact B_z: %s" % msg)
    assert np.array_equal(np.array(sim_missing.get_state("plasma")), before)
    assert sim_missing.time() == 0.0
    assert sim_missing.macro_step() == 0

    # (2) The same artifact actually executes once the qualified input datum is staged.
    sim_ok = make_sim(m, with_bz=True)
    sim_ok.install_program(compiled.so_path)
    sim_ok.step(dt)
    after = np.array(sim_ok.get_state("plasma"))
    assert np.isfinite(after).all() and not np.array_equal(after, before)
    assert sim_ok.time() == dt and sim_ok.macro_step() == 1
    print("OK  the same selected Lorentz program executes with staged B_z")

    # (3) The undeclared runtime datum is irrelevant when its operator has no selected read.
    case = Case("adc446-unused-lorentz")
    state = case.block("plasma", m)[m.states["U"]]
    unused = compile_problem(model=m, time=libtime.ForwardEuler(
        state, rate=m.operators["explicit_rhs"]))
    sim_unused = make_sim(m, with_bz=False)
    sim_unused.install_program(unused.so_path)
    sim_unused.step(dt)
    assert np.isfinite(np.array(sim_unused.get_state("plasma"))).all()
    assert sim_unused.time() == dt and sim_unused.macro_step() == 1
    print("OK  unused Lorentz declaration imposes no B_z read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
