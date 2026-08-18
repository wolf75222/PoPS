"""Public 1-d periodic scalar advection authoring for the TR-02 Gaussian pulse."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pops
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
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

from verification.pops_verify.case_authoring import resolve_case, uniform_periodic_layout

REPO_ROOT = Path(__file__).resolve().parents[4]
ADVECTION_SPEED = 1.0
CFL = 0.4


def _exact_module():
    path = Path(__file__).with_name("exact.py")
    spec = importlib.util.spec_from_file_location("tr02_gaussian_pulse_exact", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeUnavailable(RuntimeError):
    """Raised when a native TR-02 run cannot start on this machine."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case, instance, frame, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


def _frame():
    return CartesianDomain("tr02-gaussian-line", (0.0,), (1.0,)).frame(Cartesian1D())


def _author(n_cells: int) -> _Authoring:
    count = int(n_cells)
    frame = _frame()
    model = pops.Model("tr02-gaussian-advection", frame=frame)
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
    case = pops.Case("tr02-gaussian-pulse")
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


def build_case(n_cells: int):
    """Author a 1-d periodic conservative scalar advection Case."""
    return _author(n_cells).case


def resolve_plan(n_cells: int):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    authored = _author(n_cells)
    return resolve_case(
        authored.case, layout=uniform_periodic_layout(authored.frame, (authored.n_cells,))
    )


def _initial_field(n_cells: int) -> np.ndarray:
    exact = _exact_module()
    from verification.pops_verify.cell_averages import analytic_cell_averages

    count = int(n_cells)
    width = 1.0 / count
    lo = np.arange(count, dtype=np.float64) * width
    hi = lo + width
    return np.ascontiguousarray(
        analytic_cell_averages(lambda x: exact.exact_gaussian(x, 0.0, a=ADVECTION_SPEED), lo, hi)
    )


def run_native(n_cells=32, t_end=1.0, *, request=None):
    """Compile, bind, and run the 1-d Case when a C++ toolchain is present."""
    from verification.pops_verify.native_toolchain import (
        default_cxx,
        missing_compiler_requirement,
        missing_native_compile_requirement,
        repo_include,
    )
    from verification.pops_verify.native_evidence import (
        maybe_campaign_payload,
        resolution_from_request,
    )

    if request is not None and int(request.pops_native_dim) != 1:
        raise NativeUnavailable(
            f"TR-02 requires pops_native_dim=1 (got {request.pops_native_dim}); "
            "no fallback"
        )
    n_cells = resolution_from_request(request, n_cells)
    missing = missing_compiler_requirement(REPO_ROOT / "include")
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
    pops.run(simulation, t_end=float(t_end))
    field = np.ravel(np.asarray(simulation.state_global("gas"), dtype=np.float64))[
        : authored.n_cells
    ]
    return maybe_campaign_payload(
        request,
        field,
        artifact=artifact,
        simulation=simulation,
        n_cells=authored.n_cells,
        t_end=t_end,
        time_program="SSPRK2",
        cfl=CFL,
        dimension=1,
    )
