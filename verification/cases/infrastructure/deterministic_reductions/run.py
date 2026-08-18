"""IF-06 official ``pops.diagnostics`` reductions on a ConsumerGraph.

In-memory geometric / chaotic fields stay in ``exact.py`` as discrete
oracles. Native reductions are ``Integral``, ``Norm``, ``MinMax``,
``StepChangeNorm`` and ``ConservationCheck`` attached to the public
pipeline. There is no Python sequential/pairwise/blocked tree.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import (
    attach_case_diagnostics,
    bind_public,
    load_sibling_module,
    resolve_case,
    uniform_periodic_layout,
)
from verification.pops_verify.native_evidence import campaign_run_fields
from verification.pops_verify.reference_errors import reference_errors

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_v15 = load_sibling_module(Path(__file__).resolve().parents[1] / "_v15.py")

A = 1.0
CFL = 0.45
MAX_STEPS = 100_000


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


class _Authoring:
    __slots__ = ("case", "instance", "block", "frame", "n_cells")

    def __init__(
        self, case: Any, instance: Any, block: Any, frame: Any, n_cells: int
    ) -> None:
        self.case = case
        self.instance = instance
        self.block = block
        self.frame = frame
        self.n_cells = n_cells


def _sine_initial(frame):
    """Public analytic IC matching the TR-01 sine at t=0."""
    import math

    from pops.analytic import sin, x as analytic_x
    from pops.lib.initial import Analytic

    return Analytic(
        frame=frame,
        components=(1.0 + 1.0e-2 * sin(2.0 * math.pi * analytic_x(frame)),),
    )


def _frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("if06_unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


def _author(n_cells: int) -> _Authoring:
    import pops
    from pops.initial import InitialCondition
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
    frame = _frame()
    (x_axis,) = frame.axes
    model = pops.Model("if06_advection_sine", frame=frame)
    state = model.state(
        "U",
        components=("q",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (q,) = state
    velocity = model.vector("a", frame=frame, components={x_axis: A})
    flux = model.flux(
        "advection_flux",
        frame=frame,
        state=state,
        components={x_axis: (A * q,)},
        waves={x_axis: (A,)},
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
    case = pops.Case("if06_deterministic_reductions")
    tracer = case.block("tracer", model=model, states=(state,))
    instance = tracer[state]
    case.numerics(numerics, block=tracer)
    program = SSPRK2(instance, rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=CFL))
    case.program(program)
    attach_case_diagnostics(case, tracer, program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=_sine_initial(frame),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(
        case=case, instance=instance, block=tracer, frame=frame, n_cells=count
    )


def build_case(n_cells: int = _exact.N_CELLS):
    """Author the 1-d Case with official diagnostics. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = _exact.N_CELLS):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    authored = _author(n_cells)
    return resolve_case(
        authored.case,
        layout=uniform_periodic_layout(authored.frame, (authored.n_cells,)),
    )


def _native_unavailable_reason() -> str | None:
    from tests.python.support.requirements import (
        default_cxx,
        missing_compiler_requirement,
        missing_native_compile_requirement,
        repo_include,
    )

    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(n_cells: int = _exact.N_CELLS, t_end: float = 0.25, request=None):
    """Run twice through official diagnostics. Bitwise field identity is the switch."""
    import pops

    _v15.bind_campaign(request, NativeUnavailable)
    if request is not None and request.min_resolution is not None:
        n_cells = int(request.min_resolution)
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)

    def advance_once():
        authored = _author(int(n_cells))
        plan = resolve_case(
            authored.case,
            layout=uniform_periodic_layout(authored.frame, (authored.n_cells,)),
        )
        artifact = pops.compile(plan)
        mpi_mode = request.mpi_mode if request is not None else "off"
        simulation = bind_public(artifact, mpi_mode=mpi_mode)
        pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
        field = np.ravel(
            np.asarray(simulation.state_global("tracer"), dtype=np.float64)
        )
        return field[: authored.n_cells]

    try:
        first = advance_once()
        second = advance_once()
    except NativeUnavailable:
        raise
    except Exception as exc:
        raise NativeUnavailable(f"IF-06 official diagnostics run failed: {exc}") from exc
    volumes = _exact.cell_volumes(n_cells)
    errors = reference_errors(second, first, volumes)
    payload = {
        "first": first,
        "second": second,
        "linf": float(errors.linf),
        "l2": float(errors.l2),
        "diagnostics": "pops.diagnostics",
        "comparison_artifacts": {"kind": "deterministic_reductions", "linf": float(errors.linf)},
    }
    if request is None:
        return payload
    fields = campaign_run_fields(
        request=request,
        n_cells=n_cells,
        t_end=t_end,
        time_program="SSPRK2",
        cfl=CFL,
        comparison=payload["comparison_artifacts"],
    )
    fields.update(payload)
    return fields
