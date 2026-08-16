"""Typed SSPRK2/SSPRK3 Programs on genuine mono- and multi-block AMR.

Each case is authored from a symbolic :class:`pops.Case`, lowered by the public
``validate -> resolve -> compile -> bind`` lifecycle, and executed on a live two-level hierarchy.
The multi-block cells deliberately use one ordered tuple of :class:`RungeKuttaRoute` objects, so
all block stages share one Program clock and one atomic commit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pops
import pops.lib.time as libtime
import pytest
from pops.amr import (
    AMRClockRelation,
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Gaussian
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics import Model
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every


ROOT = Path(__file__).resolve().parents[4]
N = 24
DT = 5.0e-4
NSTEPS = 6
POPS_PROCESS_TIMEOUT = 900

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _transport_model():
    frame = Rectangle("typed-ssprk-amr-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    x_axis, y_axis = frame.axes
    model = Model("typed-ssprk-amr-transport", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.5 * rho * rho,), y_axis: (0.2 * rho,)},
        waves={x_axis: (rho,), y_axis: (0.2 + 0.0 * rho,)},
    )
    rate = model.rate("transport_rate", equation=ddt(state) == -div(flux))
    return frame, model, state, flux, rate


def _resolved(method: str, *, multi: bool, native_cxx: str):
    frame, model, state, flux, rate = _transport_model()
    case = pops.Case("typed-%s-amr-%s" % (method, "multi" if multi else "mono"))
    definitions = (("ions", (0.38, 0.50)), ("electrons", (0.62, 0.50)))
    rows = []
    for name, center in definitions[: 2 if multi else 1]:
        block = case.block(name, model, states=(state,))
        instance = block[state]
        numerics = DiscretizationPlan()
        numerics.rates.add(
            rate,
            FiniteVolume(
                flux=flux,
                variables=variables.Conservative(state),
                reconstruction=reconstruction.FirstOrder(),
                riemann=riemann.Rusanov(),
            ),
        )
        case.numerics(numerics, block=block)
        rows.append((name, instance, center))

    tableau = libtime.SSPRK2_TABLEAU if method == "ssprk2" else libtime.SSPRK3_TABLEAU
    if multi:
        routes = tuple(libtime.RungeKuttaRoute(instance, rate) for _name, instance, _center in rows)
        program = libtime.RungeKutta(routes=routes, tableau=tableau)
    elif method == "ssprk2":
        program = libtime.SSPRK2(rows[0][1], rate=rate)
    else:
        program = libtime.SSPRK3(rows[0][1], rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)

    x_axis, y_axis = frame.axes
    for _name, instance, center in rows:
        case.initials.add(
            InitialCondition(
                state=instance,
                value=Gaussian(
                    frame=frame,
                    center={x_axis: center[0], y_axis: center[1]},
                    background=1.0,
                    amplitude=0.4,
                    inverse_width=90.0,
                ),
                projection=ConservativeCellAverage(),
            )
        )

    threshold = case.param(RuntimeParam("refine_threshold", default=1.05))
    transfer = AMRTransfer()
    for _name, instance, _center in rows:
        transfer.state(instance, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(N, N),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(rows[0][1]) > case.value(threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(2, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": native_cxx},
    )
    return resolved, tuple(name for name, _instance, _center in rows), tableau


def _run(method: str, *, multi: bool, native_cxx: str):
    resolved, names, tableau = _resolved(method, multi=multi, native_cxx=native_cxx)
    rhs = [node for node in resolved.time.ir_nodes() if node["op"] == "rhs"]
    assert len(rhs) == len(names) * tableau.stages
    assert len(resolved.time.commits()) == len(names)

    simulation = pops.bind(pops.compile(resolved))
    assert simulation.block_names() == names
    assert simulation.n_levels() == 2
    assert simulation.patch_boxes()
    relation = simulation.program_report().level_relations
    assert relation == [
        {
            "parent_level": 0,
            "child_level": 1,
            "temporal_ratio": {"numerator": 2, "denominator": 1},
            "remainder_policy": "integral_only",
        }
    ]
    initial_mass = {name: simulation.integral(name, levels=(0,)) for name in names}
    report = pops.run(simulation, t_end=NSTEPS * DT, max_steps=NSTEPS)
    return simulation, report, initial_mass


def _assert_finite_and_conserved(simulation, report, initial_mass, label):
    assert report.accepted_steps == simulation.macro_step() == NSTEPS
    assert simulation.time() == pytest.approx(NSTEPS * DT, rel=0.0, abs=1.0e-15)
    assert simulation.patch_boxes(), "%s: no live fine patch" % label
    for name, mass_before in initial_mass.items():
        state = np.asarray(simulation.block_level_state_global(name, 0), dtype=np.float64)
        assert np.isfinite(state).all(), "%s: block %r state is not finite" % (label, name)
        absolute_drift = abs(simulation.integral(name, levels=(0,)) - mass_before)
        if mass_before == 0.0:
            assert absolute_drift < 1.0e-12, "%s: block %r zero-baseline reflux drift %.3e" % (
                label,
                name,
                absolute_drift,
            )
            continue
        relative_drift = absolute_drift / abs(mass_before)
        assert relative_drift < 1.0e-9, "%s: block %r normalized reflux drift %.3e" % (
            label,
            name,
            relative_drift,
        )


@pytest.mark.parametrize("multi", [False, True], ids=["mono", "multi"])
def test_amr_explicit_runs_finite_and_conserved(
    multi, native_cxx, isolated_native_cache, kokkos_root
):
    """The default explicit-family authority is typed SSPRK2 on real AMR."""
    del isolated_native_cache, kokkos_root
    simulation, report, mass = _run("ssprk2", multi=multi, native_cxx=native_cxx)
    _assert_finite_and_conserved(simulation, report, mass, "AMR/SSPRK2")


@pytest.mark.parametrize("multi", [False, True], ids=["mono", "multi"])
def test_amr_ssprk3_runs_finite_and_conserved(
    multi, native_cxx, isolated_native_cache, kokkos_root
):
    """Typed SSPRK3 preserves each block through live fine-patch reflux."""
    del isolated_native_cache, kokkos_root
    simulation, report, mass = _run("ssprk3", multi=multi, native_cxx=native_cxx)
    _assert_finite_and_conserved(simulation, report, mass, "AMR/SSPRK3")
