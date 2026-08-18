"""TR-06 2-d product-sine advection plus in-memory permutation identities.

``run_native`` compiles a public 2-d periodic scalar advection Case.
Permutation / reflection helpers stay exact-vs-exact utilities; they do not
write a passing case report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
CFL = 0.4
MAX_STEPS = 100_000


class NativeUnavailable(RuntimeError):
    """Raised when a native TR-06 run cannot start on this machine."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


def mapped_permutation_fields(
    n_cells: int = _exact.N_CELLS, t: float = _exact.T
):
    """Return original, swapped-coordinate, and transpose-mapped product fields."""
    x, y, volumes, _axis = _exact.uniform_grid_2d(n_cells)
    swapped_x, swapped_y = _exact.permute_xy(x, y)
    original = _exact.exact_product(
        x, y, t, kx=_exact.KX, ky=_exact.KY, ax=_exact.AX, ay=_exact.AY
    )
    at_swapped = _exact.exact_product(
        swapped_x,
        swapped_y,
        t,
        kx=_exact.KX,
        ky=_exact.KY,
        ax=_exact.AX,
        ay=_exact.AY,
    )
    mapped = at_swapped.T
    return original, at_swapped, mapped, volumes


def mapped_reflection_fields(
    n_cells: int = _exact.N_CELLS, t: float = _exact.T
):
    """Return original, x-reflected, and axis-0-flipped product fields."""
    x, y, volumes, _axis = _exact.uniform_grid_2d(n_cells)
    original = _exact.exact_product(
        x, y, t, kx=_exact.KX, ky=_exact.KY, ax=_exact.AX, ay=_exact.AY
    )
    at_reflected = _exact.exact_product(
        _exact.reflect_x(x),
        y,
        t,
        kx=_exact.KX,
        ky=_exact.KY,
        ax=_exact.AX,
        ay=_exact.AY,
    )
    mapped = at_reflected[::-1, :]
    return original, at_reflected, mapped, volumes


def permutation_linf(n_cells: int = _exact.N_CELLS, t: float = _exact.T) -> float:
    """Return L∞ of the transpose-mapped permutation identity."""
    original, _at_swapped, mapped, volumes = mapped_permutation_fields(
        n_cells, t
    )
    return float(reference_errors(mapped, original, volumes).linf)


def reflection_linf(n_cells: int = _exact.N_CELLS, t: float = _exact.T) -> float:
    """Return L∞ of the x-flipped reflection identity."""
    original, _at_reflected, mapped, volumes = mapped_reflection_fields(
        n_cells, t
    )
    return float(reference_errors(mapped, original, volumes).linf)


def _box_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D

    return CartesianDomain("tr06-box", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())


def _author(n_cells: int, *, swapped: bool = False) -> _Authoring:
    import pops
    import pops.lib.time as libtime
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.math import ddt, div
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.time import AdaptiveCFL

    count = int(n_cells)
    frame = _box_frame()
    x_axis, y_axis = frame.axes
    model = pops.Model("tr06-axis-permutation", frame=frame)
    state = model.state("U", components=("q",))
    (q,) = state
    speed_x = float(_exact.AY if swapped else _exact.AX)
    speed_y = float(_exact.AX if swapped else _exact.AY)
    velocity = model.vector(
        "a", frame=frame, components={x_axis: speed_x, y_axis: speed_y}
    )
    flux = model.flux(
        "advection_flux",
        frame=frame,
        state=state,
        components={x_axis: (speed_x * q,), y_axis: (speed_y * q,)},
        waves={x_axis: (speed_x,), y_axis: (speed_y,)},
    )
    rate = model.rate("advection_rate", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.MUSCL(limiter=limiters.VanLeer()),
            riemann=riemann.ScalarUpwind(velocity=velocity),
        ),
    )
    case = pops.Case("tr06-axis-permutation-swapped" if swapped else "tr06-axis-permutation")
    tracer = case.block("tracer", model, states=(state,))
    instance = tracer[state]
    case.numerics(numerics, block=tracer)
    program = libtime.SSPRK2(instance, rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=CFL))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(case=case, instance=instance, frame=frame, n_cells=count)


def build_case(n_cells: int = _exact.N_CELLS):
    """Author a 2-d periodic product-sine Case. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = _exact.N_CELLS):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    authored = _author(n_cells)
    return resolve_case(
        authored.case,
        layout=uniform_periodic_layout(
            authored.frame, (authored.n_cells, authored.n_cells)
        ),
    )


def _initial_field(n_cells: int, *, swapped: bool = False) -> np.ndarray:
    x, y, _volumes, _axis = _exact.uniform_grid_2d(n_cells)
    if swapped:
        field = _exact.exact_product(y, x, 0.0)
    else:
        field = _exact.exact_product(x, y, 0.0)
    return np.ascontiguousarray(field, dtype=np.float64)


def run_native(n_cells: int = _exact.N_CELLS, t_end: float = _exact.T, *, request=None):
    """Compile, bind, and run the 2-d product-sine Case."""
    import pops

    from verification.pops_verify.native_toolchain import (
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

    if request is not None and int(request.pops_native_dim) != 2:
        raise NativeUnavailable(
            f"TR-06 requires pops_native_dim=2 (got {request.pops_native_dim}); "
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
    plan = resolve_case(
        authored.case,
        layout=uniform_periodic_layout(
            authored.frame, (authored.n_cells, authored.n_cells)
        ),
    )
    artifact = pops.compile(plan)
    initial = np.ascontiguousarray(
        _initial_field(authored.n_cells)[np.newaxis, :, :], dtype=np.float64
    )
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.reshape(
        np.asarray(simulation.state_global("tracer"), dtype=np.float64),
        (authored.n_cells, authored.n_cells),
    )
    result: Any = field
    if request is not None:
        swapped = _author(n_cells, swapped=True)
        plan_p = resolve_case(
            swapped.case,
            layout=uniform_periodic_layout(
                swapped.frame, (swapped.n_cells, swapped.n_cells)
            ),
        )
        artifact_p = pops.compile(plan_p)
        initial_p = np.ascontiguousarray(
            _initial_field(swapped.n_cells, swapped=True)[np.newaxis, :, :],
            dtype=np.float64,
        )
        simulation_p = pops.bind(
            artifact_p, initial_values={swapped.instance: initial_p}
        )
        pops.run(simulation_p, t_end=float(t_end), max_steps=MAX_STEPS)
        field_p = np.reshape(
            np.asarray(simulation_p.state_global("tracer"), dtype=np.float64),
            (swapped.n_cells, swapped.n_cells),
        )
        result = {"original": field, "permuted": field_p}
    return maybe_campaign_payload(
        request,
        result,
        artifact=artifact,
        simulation=simulation,
        n_cells=authored.n_cells,
        t_end=t_end,
        time_program="SSPRK2",
        cfl=CFL,
        dimension=2,
    )
