"""TR-07 1-d discontinuous slot advection plus TV / overshoot utilities.

``run_native`` compiles a public 1-d periodic scalar advection Case.
Limiter diagnostics stay in-memory utilities; they do not write a passing
order-2 report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
ADVECTION_SPEED = float(_exact.A)
CFL = 0.4
MAX_STEPS = 100_000


class NativeUnavailable(RuntimeError):
    """Raised when a native TR-07 run cannot start on this machine."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells

OVERSHOOT_VALUE = 1.1
UNDERSHOOT_VALUE = -0.1


def total_variation(field) -> float:
    """Return the periodic 1-d total variation sum |q_{i+1}-q_i|."""
    values = np.asarray(field, dtype=np.float64).ravel()
    if values.size < 2:
        return 0.0
    jumps = np.abs(np.diff(values))
    return float(np.sum(jumps) + abs(values[0] - values[-1]))


def overshoot(field, reference) -> float:
    """Return max(0, max(field) - max(reference))."""
    sampled = np.asarray(field, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    return float(max(0.0, np.max(sampled) - np.max(exact)))


def undershoot(field, reference) -> float:
    """Return max(0, min(reference) - min(field))."""
    sampled = np.asarray(field, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    return float(max(0.0, np.min(exact) - np.min(sampled)))


def smear_slot(exact):
    """Return a three-point periodic moving average of the exact slot."""
    values = np.asarray(exact, dtype=np.float64)
    return (np.roll(values, 1) + values + np.roll(values, -1)) / 3.0


def manufactured_smeared_pair(
    n_cells: int = _exact.DEFAULT_N_CELLS,
    t=0.0,
    *,
    overshoot_value: float = OVERSHOOT_VALUE,
    undershoot_value: float = UNDERSHOOT_VALUE,
):
    """Return centers, smeared field, exact slot, and volumes."""
    centers, volumes = _exact.cell_centers(n_cells)
    reference = _exact.exact_slot(centers, t)
    field = smear_slot(reference)
    inside = np.flatnonzero(reference >= 0.5)
    outside = np.flatnonzero(reference < 0.5)
    if inside.size:
        field[int(inside[inside.size // 2])] = float(overshoot_value)
    if outside.size:
        field[int(outside[0])] = float(undershoot_value)
    return centers, field, reference, volumes


def _line_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("tr07-slot-line", (0.0,), (1.0,)).frame(Cartesian1D())


def _author(n_cells: int) -> _Authoring:
    import pops
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.lib.time import SSPRK2
    from pops.math import ddt, div
    from pops.numerics import DiscretizationPlan
    from pops.numerics.reconstruction import MUSCL
    from pops.numerics.reconstruction.limiters import VanLeer
    from pops.numerics.riemann import ScalarUpwind
    from pops.numerics.spatial import FiniteVolume
    from pops.numerics.variables import Conservative
    from pops.projection import ConservativeCellAverage
    from pops.time import FixedDt

    count = int(n_cells)
    frame = _line_frame()
    model = pops.Model("tr07-slot-advection", frame=frame)
    state = model.state("U", components=("q",))
    (q,) = state
    velocity = model.vector("a", frame=frame, components={frame.x: ADVECTION_SPEED})
    flux = model.flux(
        "F",
        frame=frame,
        state=state,
        components={frame.x: (ADVECTION_SPEED * q,)},
        waves={frame.x: (ADVECTION_SPEED,)},
    )
    rate = model.rate("explicit", equation=ddt(state) == -div(flux))
    method = FiniteVolume(
        flux=flux,
        variables=Conservative(state),
        reconstruction=MUSCL(VanLeer()),
        riemann=ScalarUpwind(velocity=velocity),
    )
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    case = pops.Case("tr07-discontinuous-slot")
    block = case.block("gas", model, states=(state,))
    instance = block[state]
    case.numerics(plan, block=block)
    program = SSPRK2(instance, rate=rate)
    program.step_strategy(FixedDt(CFL / float(count)))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(case=case, instance=instance, frame=frame, n_cells=count)


def build_case(n_cells: int = _exact.DEFAULT_N_CELLS):
    """Author a 1-d periodic slot-advection Case. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = _exact.DEFAULT_N_CELLS):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    authored = _author(n_cells)
    return resolve_case(
        authored.case, layout=uniform_periodic_layout(authored.frame, (authored.n_cells,))
    )


def _initial_field(n_cells: int) -> np.ndarray:
    centers, _volumes = _exact.cell_centers(n_cells)
    return np.ascontiguousarray(_exact.exact_slot(centers, 0.0), dtype=np.float64)


def run_native(n_cells: int = _exact.DEFAULT_N_CELLS, t_end: float = 1.0, *, request=None):
    """Compile, bind, and run the 1-d slot Case when a C++ toolchain is present."""
    import pops

    from tests.python.support.requirements import (
        default_cxx,
        missing_compiler_requirement,
        missing_native_compile_requirement,
        repo_include,
    )
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )
    from verification.pops_verify.native_evidence import (
        maybe_campaign_payload,
        resolution_from_request,
    )

    if request is not None and int(request.pops_native_dim) != 1:
        raise NativeUnavailable(
            f"TR-07 requires pops_native_dim=1 (got {request.pops_native_dim}); "
            "no fallback"
        )
    n_cells = resolution_from_request(request, n_cells)
    missing = missing_compiler_requirement(repo_include())
    if missing:
        raise NativeUnavailable(missing)
    native = missing_native_compile_requirement(repo_include(), default_cxx())
    if native:
        raise NativeUnavailable(native)
    authored = _author(n_cells)
    resolved = resolve_case(
        authored.case, layout=uniform_periodic_layout(authored.frame, (authored.n_cells,))
    )
    artifact = pops.compile(resolved)
    initial = np.ascontiguousarray(
        _initial_field(authored.n_cells)[np.newaxis, :], dtype=np.float64
    )
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.ravel(np.asarray(simulation.state_global("gas"), dtype=np.float64))[
        : authored.n_cells
    ]
    return maybe_campaign_payload(
        request,
        field,
        n_cells=authored.n_cells,
        t_end=t_end,
        time_program="SSPRK2",
        cfl=CFL,
        dimension=1,
    )
