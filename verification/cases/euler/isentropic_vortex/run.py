"""2-d periodic Euler isentropic-vortex authoring and native run.

Initial conditions are analytic cell averages of conserved fields.
``build_case`` / ``resolve_plan`` author a public 2-d periodic Euler Case
(Rusanov, WENO5-Z, SSPRK2). VanLeer is a separately labeled TVD fail
control. ``run_native`` compiles, binds, and advances the Case. A
``CampaignRequest`` returns EvidenceBundle emission keys with a packed
``(4, n, n)`` conserved result. 1-d is not applicable. Does not call ROMEO.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")

GAMMA = float(_exact.GAMMA)
N_CELLS = 32
CFL = 0.4
MAX_STEPS = 100_000
COMPONENT_ORDER = ("rho", "rho_u", "rho_v", "E")
PRIMITIVE_ORDER = ("rho", "u", "v", "p")
T_END_CANONICAL = 1.0
SPATIAL_DT_COEF = 0.16
ACCEPTANCE_RECONSTRUCTION = "weno5z"
TVD_FAIL_CONTROL_RECONSTRUCTION = "vanleer"
ACCEPTANCE_RESOLUTIONS = (16, 32, 64, 128, 256)
TVD_FAIL_CONTROL_RESOLUTIONS = (16, 32, 64, 128)


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


class _Authoring:
    __slots__ = (
        "case",
        "instance",
        "frame",
        "n_cells",
        "time_program",
        "cfl",
        "dt",
        "reconstruction",
    )

    def __init__(
        self,
        case: Any,
        instance: Any,
        frame: Any,
        n_cells: int,
        time_program: str,
        cfl: float,
        dt: float | None,
        reconstruction: str,
    ) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells
        self.time_program = time_program
        self.cfl = cfl
        self.dt = dt
        self.reconstruction = reconstruction


def reconstruction_role(name: str) -> str:
    """Acceptance is public WENO5-Z; VanLeer is a labeled TVD fail control."""
    if name == TVD_FAIL_CONTROL_RECONSTRUCTION:
        return "tvd_fail_control"
    if name == ACCEPTANCE_RECONSTRUCTION:
        return "acceptance"
    raise ValueError(f"unknown reconstruction {name!r}")


def _reconstruction_brick(name: str):
    from pops.numerics import reconstruction
    from pops.numerics.reconstruction import limiters

    if name == TVD_FAIL_CONTROL_RECONSTRUCTION:
        return reconstruction.MUSCL(limiter=limiters.VanLeer())
    if name != ACCEPTANCE_RECONSTRUCTION:
        raise ValueError(f"unknown reconstruction {name!r}")
    return reconstruction.WENO5Z()


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell-center mesh on the periodic box [0, PERIOD]^2."""
    count = int(n_cells)
    length = float(_exact.PERIOD)
    width = length / count
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    x, y = np.meshgrid(centers, centers, indexing="xy")
    return x, y, width


def cell_bounds(n_cells: int = N_CELLS):
    """Lower/upper corners of every cell, shape (n, n, 2)."""
    count = int(n_cells)
    width = float(_exact.PERIOD) / count
    axis_lo = np.arange(count, dtype=np.float64) * width
    axis_hi = axis_lo + width
    x_lo, y_lo = np.meshgrid(axis_lo, axis_lo, indexing="xy")
    x_hi, y_hi = np.meshgrid(axis_hi, axis_hi, indexing="xy")
    return np.stack((x_lo, y_lo), axis=-1), np.stack((x_hi, y_hi), axis=-1), width


def _average_scalar(fn, n_cells: int):
    from verification.pops_verify.cell_averages import analytic_cell_averages

    lo, hi, _ = cell_bounds(n_cells)
    return analytic_cell_averages(fn, lo, hi)


def average_primitives(n_cells: int, t: float = 0.0, *, u_inf=1.0, v_inf=0.0):
    """Analytic cell averages of primitive fields (rho, u, v, p)."""

    def _component(name):
        def _fn(x, y):
            return _exact.exact_vortex(x, y, t, u_inf=u_inf, v_inf=v_inf)[name]

        return _average_scalar(_fn, n_cells)

    return {name: _component(name) for name in PRIMITIVE_ORDER}


def average_conserved(n_cells: int, t: float = 0.0, *, u_inf=1.0, v_inf=0.0):
    """Analytic cell averages of conserved fields (rho, rho u, rho v, E)."""

    def _component(name):
        def _fn(x, y):
            return _exact.exact_conserved(x, y, t, u_inf=u_inf, v_inf=v_inf)[name]

        return _average_scalar(_fn, n_cells)

    return {name: _component(name) for name in COMPONENT_ORDER}


def initial_primitives(n_cells: int = N_CELLS, *, u_inf=1.0, v_inf=0.0):
    """Primitive cell averages at t=0. Each field has shape (n, n)."""
    return average_primitives(n_cells, 0.0, u_inf=u_inf, v_inf=v_inf)


def primitives_to_conserved(primitives) -> dict:
    """Convert primitive (rho, u, v, p) to conserved (rho, rho u, rho v, E)."""
    return _exact.primitives_to_conserved(primitives)


def initial_conserved(n_cells: int = N_CELLS, *, u_inf=1.0, v_inf=0.0):
    """Conserved IC: analytic cell averages of conserved fields at t=0."""
    return average_conserved(n_cells, 0.0, u_inf=u_inf, v_inf=v_inf)


def pack_conserved(conserved) -> np.ndarray:
    """Return a C-contiguous (4, n, n) array in COMPONENT_ORDER."""
    return np.ascontiguousarray(
        np.stack([conserved[name] for name in COMPONENT_ORDER], axis=0),
        dtype=np.float64,
    )


def unpack_conserved(field, n_cells: int | None = None) -> dict:
    """Split a (4, n, n) or flat native buffer into named conserved fields."""
    count = N_CELLS if n_cells is None else int(n_cells)
    array = np.ascontiguousarray(field, dtype=np.float64)
    if array.shape == (4, count, count):
        stacked = array
    else:
        stacked = np.reshape(array, (4, count, count))
    return {name: stacked[index] for index, name in enumerate(COMPONENT_ORDER)}


def conserved_to_primitives(conserved) -> dict:
    """Convert packed or named conserved fields to primitives."""
    if isinstance(conserved, np.ndarray):
        named = unpack_conserved(conserved, conserved.shape[-1])
    else:
        named = conserved
    rho = np.asarray(named["rho"], dtype=np.float64)
    momentum_x = np.asarray(named["rho_u"], dtype=np.float64)
    momentum_y = np.asarray(named["rho_v"], dtype=np.float64)
    energy = np.asarray(named["E"], dtype=np.float64)
    velocity_x = momentum_x / rho
    velocity_y = momentum_y / rho
    pressure = (GAMMA - 1.0) * (
        energy - 0.5 * rho * (velocity_x * velocity_x + velocity_y * velocity_y)
    )
    return {"rho": rho, "u": velocity_x, "v": velocity_y, "p": pressure}


def spatial_fixed_dt(n_cells: int) -> float:
    """Isolated spatial step: Δt ∝ h² (§9.4)."""
    width = float(_exact.PERIOD) / float(n_cells)
    return float(SPATIAL_DT_COEF) * width * width


def spatial_cfl(n_cells: int) -> float:
    """Per-mesh spatial CFL = Δt / h. Never the global AdaptiveCFL constant."""
    width = float(_exact.PERIOD) / float(n_cells)
    return spatial_fixed_dt(n_cells) / width


def _box_frame():
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D

    length = float(_exact.PERIOD)
    return Rectangle("eu02-box", (0.0, 0.0), (length, length)).frame(Cartesian2D())


def _author(
    n_cells: int = N_CELLS,
    *,
    dt: float | None = None,
    family: str = "global",
    reconstruction: str = ACCEPTANCE_RECONSTRUCTION,
) -> _Authoring:
    import pops
    import pops.lib.time as libtime
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.math import ddt, div, sqrt
    from pops.numerics import DiscretizationPlan, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.physics import Density, Energy, Momentum
    from pops.projection import ConservativeCellAverage
    from pops.time import AdaptiveCFL, FixedDt

    count = int(n_cells)
    frame = _box_frame()
    x_axis, y_axis = frame.axes
    model = pops.Model("eu02-euler", frame=frame)
    state = model.state(
        "U",
        components=COMPONENT_ORDER,
        roles={
            "rho": Density(),
            "rho_u": Momentum(axis=x_axis),
            "rho_v": Momentum(axis=y_axis),
            "E": Energy(),
        },
    )
    rho, momentum_x, momentum_y, energy = state
    velocity_x = momentum_x / rho
    velocity_y = momentum_y / rho
    pressure = (GAMMA - 1.0) * (
        energy - 0.5 * rho * (velocity_x * velocity_x + velocity_y * velocity_y)
    )
    sound = sqrt(GAMMA * pressure / rho)
    flux = model.flux(
        "euler",
        frame=frame,
        state=state,
        components={
            x_axis: (
                momentum_x,
                momentum_x * velocity_x + pressure,
                momentum_x * velocity_y,
                velocity_x * (energy + pressure),
            ),
            y_axis: (
                momentum_y,
                momentum_y * velocity_x,
                momentum_y * velocity_y + pressure,
                velocity_y * (energy + pressure),
            ),
        },
        waves={
            x_axis: (velocity_x - sound, velocity_x, velocity_x, velocity_x + sound),
            y_axis: (velocity_y - sound, velocity_y, velocity_y, velocity_y + sound),
        },
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux))
    case = pops.Case("eu02-isentropic-vortex")
    block = case.block("gas", model, states=(state,))
    instance = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=_reconstruction_brick(reconstruction),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    program = libtime.SSPRK2(instance, rate=rate)
    step_dt = None if dt is None else float(dt)
    if family == "spatial" and step_dt is None:
        step_dt = spatial_fixed_dt(count)
    if step_dt is not None:
        program.step_strategy(FixedDt(dt=step_dt))
        time_program = "SSPRK2+FixedDt"
        width = float(_exact.PERIOD) / float(count)
        cfl_value = (
            spatial_cfl(count) if family == "spatial" else float(step_dt / width)
        )
    else:
        program.step_strategy(AdaptiveCFL(cfl=CFL))
        time_program = "SSPRK2+AdaptiveCFL"
        cfl_value = float(CFL)
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(
        case=case,
        instance=instance,
        frame=frame,
        n_cells=count,
        time_program=time_program,
        cfl=cfl_value,
        dt=step_dt,
        reconstruction=reconstruction,
    )


def build_case(n_cells: int = N_CELLS):
    """Author a 2-d periodic gamma-law Euler Case. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = N_CELLS):
    """Validate and resolve the authored Case. Does not compile or run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells, authored.n_cells))
    return resolve_case(authored.case, layout=layout)


def _native_unavailable_reason() -> str | None:
    from verification.pops_verify.native_toolchain import (
        default_cxx,
        missing_compiler_requirement,
        missing_native_compile_requirement,
        repo_include,
    )

    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(
    n_cells: int = N_CELLS,
    t_end: float = T_END_CANONICAL,
    *,
    u_inf=1.0,
    v_inf=0.0,
    request=None,
    dt: float | None = None,
    family: str = "global",
    dump_times=None,
    reconstruction: str = ACCEPTANCE_RECONSTRUCTION,
):
    """Compile, bind, and run the 2-d vortex. Raises NativeUnavailable without Kokkos.

    ``family`` is recorded only. Constant-CFL AdaptiveCFL is ``global``.
    Isolated spatial uses ``family='spatial'`` (Δt ∝ h²). Temporal uses
    ``family='temporal'`` plus an explicit ``dt``. Default reconstruction is
    public WENO5-Z; VanLeer is a labeled TVD fail control.
    """
    import pops

    from verification.pops_verify.case_authoring import (
        bind_public,
        resolve_case,
        uniform_periodic_layout,
    )
    from verification.pops_verify.native_evidence import (
        maybe_campaign_payload,
        resolution_from_request,
    )

    if request is not None and int(request.pops_native_dim) != 2:
        raise NativeUnavailable(
            f"EU-02 requires pops_native_dim=2 (got {request.pops_native_dim}); "
            "no fallback"
        )
    n_cells = resolution_from_request(request, n_cells)
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    if family == "temporal" and dt is None:
        raise NativeUnavailable("temporal EU-02 requires an explicit FixedDt")
    authored = _author(n_cells, dt=dt, family=family, reconstruction=reconstruction)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells, authored.n_cells))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    initial = pack_conserved(initial_conserved(authored.n_cells, u_inf=u_inf, v_inf=v_inf))
    mpi_mode = getattr(request, "mpi_mode", "off") if request is not None else "off"
    simulation = bind_public(
        artifact,
        initial_values={authored.instance: initial},
        mpi_mode=mpi_mode,
    )
    snapshots: dict[str, np.ndarray] = {}
    times = []
    if dump_times:
        times = sorted({float(value) for value in dump_times})
        if float(t_end) not in times:
            times.append(float(t_end))
        times.sort()
    else:
        times = [float(t_end)]
    last = None
    for instant in times:
        pops.run(simulation, t_end=float(instant), max_steps=MAX_STEPS)
        field = np.asarray(simulation.state_global("gas"), dtype=np.float64)
        last = unpack_conserved(field, authored.n_cells)
        snapshots[f"{instant:.6g}"] = pack_conserved(last)
    if last is None:
        raise NativeUnavailable("native run produced no conserved field")
    packed = pack_conserved(last)
    output_dir = getattr(request, "output_dir", None) if request is not None else None
    if output_dir is not None and dump_times:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        np.savez(
            root / "snapshots.npz",
            times=np.asarray(times, dtype=np.float64),
            **{f"t_{index}": snapshots[f"{instant:.6g}"] for index, instant in enumerate(times)},
        )
    if request is None:
        return last
    extra: dict[str, Any] = {}
    if authored.dt is not None:
        extra["dt"] = float(authored.dt)
    return maybe_campaign_payload(
        request,
        packed,
        artifact=artifact,
        simulation=simulation,
        n_cells=authored.n_cells,
        t_end=t_end,
        time_program=authored.time_program,
        cfl=authored.cfl,
        dimension=2,
        **extra,
    )
