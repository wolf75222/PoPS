"""Public 1-d periodic cold-electron Euler–Poisson authoring and native run.

SSPRK2 is wired with ``fields=`` so Poisson is solved at each stage (TM-07).
``run_native`` compiles, binds, and advances the Case. Fluid |u| is not the
Langmuir phase speed; the step is ``FixedDt`` from ω_pe / k.

Acceptance reconstruction is public WENO5-Z. MUSCL+VanLeer is a labeled
non-acceptance TVD variant. Global series keep constant phase CFL; temporal
and isolated spatial (``dt ∝ h²``) campaigns are separately labeled.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.campaign import CampaignJob, CampaignRequest, job_to_dict
from verification.pops_verify.case_authoring import bind_public, load_sibling_module
from verification.pops_verify.cell_averages import analytic_cell_averages
from verification.pops_verify.native_evidence import (
    apply_campaign_request,
    maybe_campaign_payload,
    require_bind_request,
    run_fields_from_payload,
)

_CASE_DIR = Path(__file__).resolve().parent
_EXACT = load_sibling_module(_CASE_DIR / "exact.py")

N_CELLS = _EXACT.N_CELLS
E_CHARGE = _EXACT.E_CHARGE
Q_E = _EXACT.Q_E
N_I = _EXACT.N_I
EPS0 = _EXACT.EPS0
M_E = _EXACT.M_E
PHASE_CFL = _EXACT.PHASE_CFL
RESOLUTIONS = (16, 32, 64, 128)
TEMPORAL_N = 256
MAX_STEPS = 100_000
PROBE_X = 0.25
SAMPLES_PER_PERIOD = 64
FREQUENCY_PERIODS = 4
DEFAULT_RECONSTRUCTION = "weno5z"
A = _EXACT.A
K = _EXACT.K
N0 = _EXACT.N0


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


class AuthoringPending(RuntimeError):
    """Kept for compatibility. Resolve now succeeds with SSPRK2(fields=...)."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells", "dt")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int, dt: float) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells
        self.dt = float(dt)


def period() -> float:
    return 2.0 * np.pi / _EXACT.plasma_frequency()


def phase_dt(n_cells: int, cfl: float = PHASE_CFL) -> float:
    """Δt from the Langmuir phase speed ω_pe / k, not the O(A) fluid speed."""
    dx = 1.0 / float(n_cells)
    phase_speed = _EXACT.plasma_frequency() / _EXACT.K
    return float(cfl) * dx / phase_speed


def phase_dt_h2(n_cells: int, reference_n: int = 16) -> float:
    """Isolated spatial step: dt(n) = dt_phase(N_ref) * (N_ref / n)²."""
    return phase_dt(int(reference_n)) * (float(reference_n) / float(n_cells)) ** 2


def default_temporal_dts(n_cells: int = TEMPORAL_N) -> tuple[float, ...]:
    """Fixed-grid temporal series chosen so temporal error dominates.

    Phase-CFL dt on n=256 is already spatially saturated. Coarsest dt is
    T/16 (independent of n) then halved three times.
    """
    del n_cells
    dt0 = period() / 16.0
    return tuple(dt0 / (2 ** index) for index in range(4))


def _reconstruction_brick(name: str):
    from pops.numerics import reconstruction
    from pops.numerics.reconstruction import limiters

    token = str(name or DEFAULT_RECONSTRUCTION)
    if token == "vanleer":
        return reconstruction.MUSCL(limiter=limiters.VanLeer())
    if token not in {"weno5z", "weno5"}:
        raise ValueError(f"unsupported reconstruction {token!r}")
    return reconstruction.WENO5Z()


def fields_from_density(density):
    """Spectral Gauss/Poisson fields consistent with the conserved density IC."""
    samples = np.asarray(density, dtype=np.float64)
    count = int(samples.size)
    spacing = 1.0 / float(count)
    rhs = E_CHARGE * (N_I - samples) / EPS0
    wave = 2.0 * np.pi * np.fft.fftfreq(count, d=spacing)
    rhs_hat = np.fft.fft(rhs)
    phi_hat = np.zeros_like(rhs_hat)
    electric_hat = np.zeros_like(rhs_hat)
    nonzero = np.abs(wave) > 0.0
    phi_hat[nonzero] = rhs_hat[nonzero] / (wave[nonzero] * wave[nonzero])
    electric_hat[nonzero] = rhs_hat[nonzero] / (1j * wave[nonzero])
    return (
        np.ascontiguousarray(np.fft.ifft(phi_hat).real, dtype=np.float64),
        np.ascontiguousarray(np.fft.ifft(electric_hat).real, dtype=np.float64),
    )


def energy_baseline(ke, ese) -> dict[str, Any]:
    """Total-energy conservation after skipping a leading fake (KE, ESE)=(0,0).

    Cold Langmuir is a standing wave: KE and ESE exchange. That oscillation is
    not conservation drift. Drift is reported on total energy only.
    """
    kinetic = [float(value) for value in ke]
    electrostatic = [float(value) for value in ese]
    energy = [a + b for a, b in zip(kinetic, electrostatic)]
    start = 0
    while start < len(energy) and energy[start] == 0.0:
        start += 1
    if start >= len(energy):
        return {
            "initial": None,
            "final": energy[-1] if energy else None,
            "max_relative_drift": None,
            "ese_oscillation": None,
            "ke_oscillation": None,
            "conservation": "total_energy",
            "values": energy,
        }
    baseline = energy[start]
    rest = energy[start:]
    drift = None
    if baseline != 0.0:
        drift = float(max(abs(value - baseline) for value in rest) / abs(baseline))
    ese_rest = electrostatic[start:]
    ke_rest = kinetic[start:]
    return {
        "initial": float(baseline),
        "final": float(rest[-1]),
        "max_relative_drift": drift,
        "ese_oscillation": float(max(ese_rest) - min(ese_rest)) if ese_rest else None,
        "ke_oscillation": float(max(ke_rest) - min(ke_rest)) if ke_rest else None,
        "conservation": "total_energy",
        "values": rest,
    }


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell centers on the periodic unit interval."""
    return _EXACT.uniform_cell_centers(n_cells)


def _cell_bounds(n_cells: int):
    width = 1.0 / float(n_cells)
    lo = np.arange(int(n_cells), dtype=np.float64) * width
    return lo, lo + width, np.full(int(n_cells), width, dtype=np.float64)


def cell_average_fields(n_cells: int, t: float = 0.0) -> dict[str, np.ndarray]:
    """§7.3 cell averages of the closed oracle. Not point samples."""
    count = int(n_cells)
    time = float(t)
    lo, hi, volumes = _cell_bounds(count)
    density = analytic_cell_averages(lambda x: _EXACT.n_e(x, time), lo, hi)
    velocity = analytic_cell_averages(lambda x: _EXACT.u_e(x, time), lo, hi)
    momentum = analytic_cell_averages(
        lambda x: _EXACT.n_e(x, time) * _EXACT.u_e(x, time), lo, hi
    )
    electric = analytic_cell_averages(lambda x: _EXACT.e_field(x, time), lo, hi)
    potential = analytic_cell_averages(lambda x: _EXACT.phi(x, time), lo, hi)
    return {
        "n": density,
        "u": velocity,
        "nu": momentum,
        "e": electric,
        "phi": potential,
        "volumes": volumes,
    }


def initial_fields(n_cells: int = N_CELLS, t: float = 0.0):
    """Closed-form n_e, u_e, E, φ at cell centers (diagnostics only)."""
    centers, volumes = cell_centers(n_cells)
    time = float(t)
    return {
        "x": centers,
        "volumes": volumes,
        "n_e": _EXACT.n_e(centers, time),
        "u_e": _EXACT.u_e(centers, time),
        "e": _EXACT.e_field(centers, time),
        "phi": _EXACT.phi(centers, time),
    }


def initial_conserved(n_cells: int = N_CELLS):
    """Conserved IC (n, n u) from cell averages of the closed oracle at t=0."""
    sample = cell_average_fields(n_cells, 0.0)
    return np.stack((sample["n"], sample["nu"]))


def _spectral_dx(field, length: float = 1.0) -> np.ndarray:
    samples = np.asarray(field, dtype=np.float64)
    spacing = float(length) / float(samples.size)
    wave = 2.0 * np.pi * np.fft.fftfreq(samples.size, d=spacing)
    return np.fft.ifft(1j * wave * np.fft.fft(samples)).real


def _line_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("cp02-line", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())


def _analytic_initial(frame):
    from pops.analytic import constant, cos, x as analytic_x
    from pops.lib.initial import Analytic

    xx = analytic_x(frame)
    density = float(N0) + float(A) * cos(float(K) * xx)
    return Analytic(frame=frame, components=(density, constant(0.0)))


def _author(
    n_cells: int = N_CELLS,
    dt: float | None = None,
    reconstruction: str = DEFAULT_RECONSTRUCTION,
) -> _Authoring:
    import pops
    from pops.fields import (
        CellCenteredSecondOrder,
        ConstantNullspace,
        FieldDiscretization,
        FieldOutput,
        GradientOutput,
        MeanValueGauge,
    )
    from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
    from pops.initial import InitialCondition
    from pops.lib.time import SSPRK2
    from pops.math import ddt, div, laplacian
    from pops.numerics import DiscretizationPlan, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.physics import Density, Momentum
    from pops.projection import ConservativeCellAverage
    from pops.solvers.elliptic import FFT
    from pops.time import FixedDt

    count = int(n_cells)
    step = float(dt if dt is not None else phase_dt(count))
    frame = _line_frame()
    (x_axis,) = frame.axes
    model = pops.Model("cp02-langmuir-cold", frame=frame)
    state = model.state(
        "U",
        components=("n", "n_u"),
        roles={
            "n": Density(),
            "n_u": Momentum(axis=x_axis),
        },
    )
    density, momentum = state
    velocity = momentum / density
    flux = model.flux(
        "cold_electron",
        frame=frame,
        state=state,
        components={x_axis: (momentum, momentum * velocity)},
        waves={x_axis: (velocity, velocity)},
    )
    potential = model.field("phi")
    phi_aux = model.aux("potential")
    electric = model.aux("phi_grad_x")
    charge = model.source(
        "electric",
        on=state,
        value=(0.0 * density + 0.0 * phi_aux, (Q_E / M_E) * density * electric),
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux) + charge)
    operator = model.field_operator(
        "fields",
        unknown=potential,
        equation=(-laplacian(potential) == (E_CHARGE / EPS0) * (N_I - density)),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("phi_grad", potential, sign=-1),
        ),
    )
    case = pops.Case("cp02-langmuir-cold")
    block = case.block("electrons", model, states=(state,))
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
    field = case.field(
        operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
            solver=FFT(),
            nullspace=ConstantNullspace(),
            gauge=MeanValueGauge(0.0),
        ),
    )
    program = SSPRK2(instance, rate=rate, fields=field)
    program.step_strategy(FixedDt(step))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=_analytic_initial(frame),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(case=case, instance=instance, frame=frame, n_cells=count, dt=step)


def build_case(n_cells: int = N_CELLS):
    """Author a 1-d periodic cold Euler–Poisson Case. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = N_CELLS):
    """Validate and resolve the Case. Does not compile or execute a run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    return resolve_case(authored.case, layout=layout)


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


def _live_runtime_facts() -> dict[str, Any]:
    try:
        from pops.runtime_environment import runtime_environment_report

        return dict(runtime_environment_report())
    except Exception:
        return {
            "kokkos_backend": "unknown",
            "kokkos_concurrency": 0,
            "kokkos_initialized": False,
        }


def _space_from_backend(backend: str, requested: str) -> str:
    token = str(backend)
    if token in {"OpenMP", "KokkosOpenMP"}:
        return "KokkosOpenMP"
    if token in {"Serial", "KokkosSerial"}:
        return "KokkosSerial"
    if token in {"Cuda", "KokkosCuda"}:
        return "KokkosCuda"
    return requested


def _require_live_space(request) -> str:
    facts = _live_runtime_facts()
    backend = str(facts.get("kokkos_backend") or "unknown")
    requested = str(getattr(request, "execution_space", None) or "KokkosSerial")
    if requested == "KokkosSerial" and backend in {
        "OpenMP",
        "KokkosOpenMP",
        "Cuda",
        "KokkosCuda",
    }:
        raise NativeUnavailable(
            f"KokkosSerial requested but live backend is {backend}; Serial not-run"
        )
    if requested == "KokkosOpenMP" and backend not in {"OpenMP", "KokkosOpenMP", "unknown"}:
        raise NativeUnavailable(f"KokkosOpenMP requested but live backend is {backend}")
    if requested == "KokkosCuda" and backend not in {"Cuda", "KokkosCuda"}:
        raise NativeUnavailable(f"KokkosCuda requested but live backend is {backend}")
    return requested


def _truthful_overrides(request, n_cells: int, t_end: float, cfl: float) -> dict[str, Any]:
    from tests.python.support.requirements import default_cxx
    from verification.pops_verify.mpi_world import native_world_size

    mpi_on = getattr(request, "mpi_mode", "off") == "on"
    facts = _live_runtime_facts()
    live_ranks = native_world_size(required=False)
    if live_ranks is not None:
        mpi_ranks = int(live_ranks)
    elif mpi_on:
        mpi_ranks = int(getattr(request.resources, "mpi_ranks", None) or 1)
    else:
        mpi_ranks = 1
    concurrency = int(facts.get("kokkos_concurrency") or 0)
    threads = concurrency if concurrency > 0 else int(
        getattr(request.resources, "omp_threads", None) or 1
    )
    return {
        "compiler": default_cxx() or os.environ.get("CXX") or "unknown",
        "build_type": os.environ.get("CMAKE_BUILD_TYPE") or "Release",
        "mpi_ranks": mpi_ranks,
        "omp_threads_per_rank": threads,
        "kokkos_execution_space": _space_from_backend(
            str(facts.get("kokkos_backend") or "unknown"),
            str(getattr(request, "execution_space", None) or "KokkosSerial"),
        ),
        "mpi_library": (os.environ.get("POPS_MPI_LIBRARY") or "unknown") if mpi_on else "none",
        "mpi_thread_level_requested": "MPI_THREAD_SINGLE" if mpi_on else "none",
        "mpi_thread_level_provided": "MPI_THREAD_SINGLE" if mpi_on else "none",
    }


def _potential(simulation, n_cells: int) -> np.ndarray:
    slots = tuple(simulation.field_provider_slots())
    if not slots:
        raise NativeUnavailable("native runtime exposed no field-provider slot")
    phi = np.ravel(np.asarray(simulation.field_potential_global(slots[0]), dtype=np.float64))
    if phi.size < int(n_cells):
        raise NativeUnavailable("field potential length is shorter than the mesh")
    return np.ascontiguousarray(phi[: int(n_cells)], dtype=np.float64)


def _phi_amplitude() -> float:
    return abs(E_CHARGE * A / (EPS0 * K * K))


def _solve_initial_fields(simulation) -> None:
    action = getattr(simulation, "solve_fields", None)
    if callable(action):
        action()
        return
    executor = getattr(simulation, "_executor", None)
    action = getattr(executor, "solve_fields", None) if executor is not None else None
    if callable(action):
        action()


def _snapshot(simulation, authored: _Authoring, time: float) -> dict[str, Any]:
    field = np.reshape(
        np.asarray(simulation.state_global("electrons"), dtype=np.float64),
        (2, authored.n_cells),
    )
    try:
        phi = _potential(simulation, authored.n_cells)
    except NativeUnavailable:
        phi = np.zeros(authored.n_cells, dtype=np.float64)
    if float(np.max(np.abs(phi))) < 0.25 * _phi_amplitude():
        phi, electric = fields_from_density(field[0])
    else:
        electric = -_spectral_dx(phi)
    dx = 1.0 / float(authored.n_cells)
    density = field[0]
    velocity = field[1] / np.maximum(density, 1.0e-30)
    index = min(int(PROBE_X * authored.n_cells), authored.n_cells - 1)
    return {
        "time": float(time),
        "field": np.ascontiguousarray(field, dtype=np.float64),
        "phi": phi,
        "electric": np.ascontiguousarray(electric, dtype=np.float64),
        "mass": float(np.sum(density) * dx),
        "momentum": float(np.sum(field[1]) * dx),
        "charge": float(np.sum(E_CHARGE * (N_I - density)) * dx),
        "ke": float(np.sum(0.5 * M_E * density * velocity * velocity) * dx),
        "ese": float(np.sum(0.5 * EPS0 * electric * electric) * dx),
        "probe_n": float(density[index]),
        "probe_e": float(electric[index]),
    }


def _coupling_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    final = records[-1]
    return {
        "times": [float(item["time"]) for item in records],
        "probe_n": [float(item["probe_n"]) for item in records],
        "probe_e": [float(item["probe_e"]) for item in records],
        "mass": [float(item["mass"]) for item in records],
        "momentum": [float(item["momentum"]) for item in records],
        "charge": [float(item["charge"]) for item in records],
        "ke": [float(item["ke"]) for item in records],
        "ese": [float(item["ese"]) for item in records],
        "phi": final["phi"].tolist(),
        "electric": final["electric"].tolist(),
        "snapshots_n": [item["field"][0].tolist() for item in records],
    }


def default_sample_times(t_end: float) -> np.ndarray:
    horizon = float(t_end)
    count = int(SAMPLES_PER_PERIOD)
    interior = np.linspace(horizon / float(count), horizon, count)
    return np.ascontiguousarray(interior, dtype=np.float64)


def final_order_sample_times(t_end: float) -> np.ndarray:
    """Single horizon stamp so FixedDt is not clipped by a 64-point grid."""
    return np.ascontiguousarray([float(t_end)], dtype=np.float64)


def expected_accepted_steps(t_end: float, dt: float) -> int:
    return int(round(float(t_end) / float(dt)))


def stamp_grid_clips_dt(stamps, dt) -> bool:
    grid = np.asarray(stamps, dtype=np.float64)
    if grid.size == 0:
        return True
    if grid.size == 1:
        return False
    starts = np.concatenate((np.asarray([0.0], dtype=np.float64), grid[:-1]))
    return bool(np.any(grid - starts + 1.0e-15 < float(dt)))


def sample_times_for_family(family: str, t_end: float, sample_times=None) -> np.ndarray:
    if sample_times is not None:
        return np.asarray(sample_times, dtype=np.float64)
    if str(family) == "temporal":
        return final_order_sample_times(t_end)
    return default_sample_times(t_end)


def result_array_digest(field) -> str:
    array = np.ascontiguousarray(np.asarray(field, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def run_native(
    n_cells: int = N_CELLS,
    t_end: float | None = None,
    *,
    request=None,
    dt: float | None = None,
    sample_times=None,
    reconstruction: str | None = None,
    family: str = "global",
):
    """Compile, bind, and run the Langmuir case. Raises NativeUnavailable without Kokkos."""
    n_cells = apply_campaign_request(
        n_cells, request, case_id="CP-02", allowed_dims=(1,), unavailable=NativeUnavailable
    )
    import pops

    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    horizon = float(t_end if t_end is not None else period())
    step = float(dt if dt is not None else phase_dt(n_cells))
    recon = str(reconstruction or DEFAULT_RECONSTRUCTION)
    authored = _author(n_cells, dt=step, reconstruction=recon)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    mpi_mode = require_bind_request(request, NativeUnavailable, "CP-02")
    if request is not None:
        _require_live_space(request)
        if mpi_mode == "on":
            from verification.pops_verify.mpi_world import native_world_size

            ranks = native_world_size(required=False)
            if ranks is None or int(ranks) < 2:
                raise NativeUnavailable(
                    "CP-02 MPI requested but native world has fewer than 2 ranks"
                )
    simulation = bind_public(artifact, mpi_mode=mpi_mode)
    try:
        _solve_initial_fields(simulation)
    except Exception:
        pass
    stamps = sample_times_for_family(family, horizon, sample_times)
    if stamps.ndim != 1 or stamps.size == 0:
        raise ValueError("sample_times must be a non-empty 1-d array")
    records = [_snapshot(simulation, authored, 0.0)]
    accepted = 0
    if str(family) == "temporal":
        report = pops.run(simulation, t_end=horizon, max_steps=MAX_STEPS)
        accepted = int(getattr(report, "accepted_steps", 0) or 0)
        records.append(_snapshot(simulation, authored, horizon))
    else:
        for stamp in stamps:
            report = pops.run(simulation, t_end=float(stamp), max_steps=MAX_STEPS)
            accepted += int(getattr(report, "accepted_steps", 0) or 0)
            records.append(_snapshot(simulation, authored, float(stamp)))
    field = records[-1]["field"]
    if request is None:
        return field
    payload = maybe_campaign_payload(
        request,
        field,
        n_cells=n_cells,
        t_end=horizon,
        time_program="SSPRK2",
        cfl=float(PHASE_CFL),
        dimension=1,
        artifact=artifact,
        simulation=simulation,
        coupling=_coupling_from_records(records),
        **_truthful_overrides(request, n_cells, horizon, PHASE_CFL),
    )
    payload["dt"] = authored.dt
    payload["family"] = str(family)
    payload["reconstruction"] = recon
    payload["accepted_steps"] = accepted
    payload["expected_accepted_steps"] = expected_accepted_steps(horizon, authored.dt)
    payload["result_digest"] = result_array_digest(field)
    return payload


def run_native_series(times, n_cells: int = N_CELLS):
    """Advance one bound run through increasing ``times``. Shape (len(times), 2, n)."""
    stamps = np.asarray(times, dtype=np.float64)
    if stamps.ndim != 1 or stamps.size == 0:
        raise ValueError("times must be a non-empty 1-d array")
    if np.any(np.diff(stamps) <= 0.0):
        raise ValueError("times must be strictly increasing")
    import pops

    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    simulation = bind_public(artifact, mpi_mode="off")
    try:
        _solve_initial_fields(simulation)
    except Exception:
        pass
    snapshots = np.empty((stamps.size, 2, authored.n_cells), dtype=np.float64)
    for index, stamp in enumerate(stamps):
        pops.run(simulation, t_end=float(stamp), max_steps=MAX_STEPS)
        field = np.asarray(simulation.state_global("electrons"), dtype=np.float64)
        snapshots[index] = np.reshape(field, (2, authored.n_cells))
    return snapshots


def _emit_job(job_dir: Path, payload: dict[str, Any], request, identity, n_cells: int) -> Path:
    from verification.pops_verify.evidence_contract import emit_job_directory
    from verification.pops_verify.metrics import collect_metrics
    from verification.pops_verify.provenance import collect_provenance

    job = CampaignJob(
        case_id="CP-02",
        pops_native_dim=1,
        suite=getattr(request, "suite", "pr"),
        execution_space=str(payload.get("kokkos_execution_space") or request.execution_space),
        mpi_mode=request.mpi_mode,
        min_resolution=int(n_cells),
        resources=request.resources,
        evidence_status=getattr(request, "evidence_status", "required"),
    )
    job_dict = job_to_dict(job)
    if payload.get("dt") is not None:
        job_dict["dt"] = float(payload["dt"])
    if payload.get("accepted_steps") is not None:
        job_dict["accepted_steps"] = int(payload["accepted_steps"])
    if payload.get("expected_accepted_steps") is not None:
        job_dict["expected_accepted_steps"] = int(payload["expected_accepted_steps"])
    if payload.get("result_digest") is not None:
        job_dict["result_digest"] = str(payload["result_digest"])
    resolved = {
        "case": {"id": "CP-02"},
        "job": job_dict,
        "status": "run",
        "reason": None,
        "family": payload.get("family", "global"),
        "reconstruction": payload.get("reconstruction", DEFAULT_RECONSTRUCTION),
    }
    provenance = collect_provenance(
        "CP-02",
        pops_native_dim=1,
        dimension=1,
        nodes=int(getattr(request.resources, "nodes", None) or 1),
        pops_version=None,
        doctor_ok=bool(identity.doctor_ok),
        component_catalog_digest=identity.component_catalog_digest,
        native_header_signature=identity.native_header_signature,
        native_variant_manifest_digest=identity.native_variant_manifest_digest,
        **run_fields_from_payload(payload),
    )
    metrics = collect_metrics("CP-02", reason="per-job payload; series analysis fills errors")
    extra = {
        "result": payload["result"],
        "program_bytes": payload["program_bytes"],
        "coupling": payload.get("coupling"),
    }
    return emit_job_directory(
        job_dir,
        resolved_case=resolved,
        provenance=provenance,
        metrics=metrics,
        result=extra["result"],
        program_bytes=extra["program_bytes"],
        native_artifact={
            "path": str(identity.path),
            "sha256": identity.sha256,
            "dimension": identity.dimension,
        },
        coupling=extra.get("coupling"),
    )


def _authenticate_request(request):
    if request is None:
        raise NativeUnavailable("CP-02 campaign requires CampaignRequest")
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    from verification.pops_verify.capabilities import authenticate_installed_artifact

    try:
        identity = authenticate_installed_artifact(dimension=1, doctor_ok=False)
    except Exception as exc:
        raise NativeUnavailable(f"CP-02 has no authenticated Dim1 leaf: {exc}") from exc
    if request.execution_space == "KokkosOpenMP" and not identity.has_kokkos:
        raise NativeUnavailable("CP-02 OpenMP requested but the installed leaf has no Kokkos")
    if request.mpi_mode == "on" and not identity.has_mpi:
        raise NativeUnavailable("CP-02 MPI requested but the installed leaf has no MPI")
    return identity


def run_order_campaign(
    output_dir,
    *,
    request=None,
    resolutions: tuple[int, ...] = RESOLUTIONS,
    t_end: float | None = None,
    reconstruction: str | None = None,
    family: str = "global",
    dt_fn=None,
):
    """Run ≥4 resolutions and emit EvidenceBundle-compatible job/series files."""
    from verification.pops_verify.evidence_contract import write_series_json

    identity = _authenticate_request(request)
    horizon = float(t_end if t_end is not None else FREQUENCY_PERIODS * period())
    recon = str(reconstruction or DEFAULT_RECONSTRUCTION)
    counts = tuple(int(item) for item in resolutions)
    if len(counts) < 1:
        raise NativeUnavailable("CP-02 campaign requires at least one resolution")
    series_root = Path(output_dir)
    jobs: list[str] = []
    for count in counts:
        job_request = CampaignRequest(
            case_id="CP-02",
            pops_native_dim=1,
            suite=request.suite,
            execution_space=request.execution_space,
            mpi_mode=request.mpi_mode,
            min_resolution=count,
            resources=request.resources,
            evidence_status=request.evidence_status,
            output_dir=request.output_dir,
        )
        step = None if dt_fn is None else float(dt_fn(count))
        payload = run_native(
            count,
            horizon,
            request=job_request,
            dt=step,
            reconstruction=recon,
            family=family,
        )
        name = f"n{count}"
        _emit_job(series_root / name, payload, job_request, identity, count)
        jobs.append(name)
    write_series_json(series_root, "CP-02", jobs)
    return series_root


def run_temporal_campaign(
    output_dir,
    *,
    request=None,
    n_cells: int = TEMPORAL_N,
    dts=None,
    t_end: float | None = None,
    reconstruction: str | None = None,
):
    """Separated temporal study: fixed fine N, dt / dt/2 / dt/4 / dt/8."""
    from verification.pops_verify.evidence_contract import write_series_json

    identity = _authenticate_request(request)
    horizon = float(t_end if t_end is not None else period())
    recon = str(reconstruction or DEFAULT_RECONSTRUCTION)
    steps = tuple(float(value) for value in (dts if dts is not None else default_temporal_dts(n_cells)))
    if len(steps) < 4:
        raise NativeUnavailable("CP-02 temporal campaign requires at least four dt values")
    series_root = Path(output_dir)
    jobs: list[str] = []
    job_request = CampaignRequest(
        case_id="CP-02",
        pops_native_dim=1,
        suite=request.suite,
        execution_space=request.execution_space,
        mpi_mode=request.mpi_mode,
        min_resolution=int(n_cells),
        resources=request.resources,
        evidence_status=request.evidence_status,
        output_dir=request.output_dir,
    )
    for index, step in enumerate(steps):
        payload = run_native(
            int(n_cells),
            horizon,
            request=job_request,
            dt=step,
            reconstruction=recon,
            family="temporal",
        )
        name = f"dt{index}"
        _emit_job(series_root / name, payload, job_request, identity, int(n_cells))
        jobs.append(name)
    write_series_json(series_root, "CP-02", jobs)
    return series_root


def run_spatial_campaign(
    output_dir,
    *,
    request=None,
    resolutions: tuple[int, ...] = RESOLUTIONS,
    t_end: float | None = None,
    reconstruction: str | None = None,
):
    """Isolated spatial series with dt ∝ h². Labeled spatial, not global."""
    return run_order_campaign(
        output_dir,
        request=request,
        resolutions=resolutions,
        t_end=t_end,
        reconstruction=reconstruction,
        family="spatial",
        dt_fn=phase_dt_h2,
    )
