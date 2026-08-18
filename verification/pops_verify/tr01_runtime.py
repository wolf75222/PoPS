"""Shared 1-d TR-01 compile / bind / run for IF and PF campaigns.

Authors a dedicated 1-d periodic sine Case. The canonical TR-01 catalog
entry in ``transport/advection_sine`` stays 3-d. ``pops.run`` stays here
so case modules can call this helper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import (
    load_sibling_module,
    resolve_case,
    uniform_periodic_layout,
)

_REPO = Path(__file__).resolve().parents[2]
_TR01 = _REPO / "verification" / "cases" / "transport" / "advection_sine" / "run.py"
_TR01_EXACT = _TR01.with_name("exact.py")
A_1D = 1.0
CFL_1D = 0.45
CASE_NAME = "tr01_advection_sine"


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


class Prepared:
    __slots__ = ("module", "authored", "artifact", "initial")

    def __init__(self, module: Any, authored: Any, artifact: Any, initial: np.ndarray) -> None:
        self.module = module
        self.authored = authored
        self.artifact = artifact
        self.initial = initial


def tr01_module():
    return load_sibling_module(_TR01)


def author(n_cells: int) -> _Authoring:
    """Author the shared 1-d periodic sine Case used by IF / PF siblings."""
    import pops
    from pops.analytic import sin, x as analytic_x
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D
    from pops.initial import InitialCondition
    from pops.lib.initial import Analytic
    from pops.lib.time import SSPRK2
    from pops.math import ddt, div
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.representations import Conservative
    from pops.spaces import CellState
    from pops.time import AdaptiveCFL

    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    exact = load_sibling_module(_TR01_EXACT)
    frame = CartesianDomain("unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())
    (x_axis,) = frame.axes
    model = pops.Model(CASE_NAME, frame=frame)
    state = model.state(
        "U",
        components=("q",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (q,) = state
    velocity = model.vector("a", frame=frame, components={x_axis: A_1D})
    flux = model.flux(
        "advection_flux",
        frame=frame,
        state=state,
        components={x_axis: (A_1D * q,)},
        waves={x_axis: (A_1D,)},
    )
    rate = model.rate("advection_rate", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
            riemann=riemann.ScalarUpwind(velocity=velocity),
        ),
    )
    case = pops.Case(CASE_NAME)
    tracer = case.block("tracer", model=model, states=(state,))
    instance = tracer[state]
    case.numerics(numerics, block=tracer)
    program = SSPRK2(instance, rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=CFL_1D))
    case.program(program)
    wave = 2.0 * np.pi * 1.0
    case.initials.add(
        InitialCondition(
            state=instance,
            value=Analytic(
                frame=frame,
                components=(
                    float(exact.Q0) + float(exact.EPS) * sin(wave * analytic_x(frame)),
                ),
            ),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(case=case, instance=instance, frame=frame, n_cells=count)


def build_case(n_cells: int):
    """Return the dedicated 1-d Case. Does not compile or run."""
    return author(int(n_cells)).case


def prepare(n_cells: int, *, consumers=None, attach=None) -> Prepared:
    """Validate/compile the 1-d TR-01 helper. Optional ``attach`` or consumers."""
    import pops

    module = tr01_module()
    missing = module._native_unavailable_reason()
    if missing:
        raise module.NativeUnavailable(missing)
    authored = author(int(n_cells))
    if attach is not None:
        attach(authored)
    if consumers is not None:
        authored.case.consumers(consumers)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    exact = load_sibling_module(_TR01_EXACT)
    centers, _ = exact.uniform_cell_centers(authored.n_cells)
    initial = np.ascontiguousarray(
        exact.exact_sine(centers, 0.0)[np.newaxis, :],
        dtype=np.float64,
    )
    return Prepared(module, authored, artifact, initial)


def advance(prepared: Prepared, t_end: float, *, output_dir=None) -> np.ndarray:
    """Bind and run one prepared artifact. Returns the gathered 1-d field."""
    import pops

    from verification.pops_verify.case_authoring import bind_public

    simulation = bind_public(
        prepared.artifact,
        initial_values={prepared.authored.instance: prepared.initial},
    )
    kwargs = {"t_end": float(t_end), "max_steps": prepared.module.MAX_STEPS}
    if output_dir is not None:
        kwargs["output_dir"] = output_dir
    pops.run(simulation, **kwargs)
    return np.ravel(np.asarray(simulation.state_global("tracer"), dtype=np.float64))
