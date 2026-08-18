"""TR-01 public 3-d oblique periodic sine advection.

Annexe A.1 / §35.1 ``01_advection_sine_oblique_3d``: a = (1, 1, 1),
k = (1, 2, 3), T = 1 on the periodic unit cube. Requires a native
artifact compiled with ``POPS_NATIVE_DIM=3``. There is no 1-d or 2-d
fallback.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pops
from pops.analytic import sin, x as analytic_x, y as analytic_y, z as analytic_z
from pops.domain import CartesianDomain
from pops.frames import Cartesian3D
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
from tests.python.support.requirements import (
    default_cxx,
    missing_compiler_requirement,
    missing_native_compile_requirement,
    repo_include,
)
from verification.pops_verify.case_authoring import (
    attach_case_diagnostics,
    bind_public,
    load_sibling_module,
    resolve_case,
    uniform_periodic_layout,
)
from verification.pops_verify.cell_averages import analytic_cell_averages
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.provenance import collect_provenance, write_provenance
from verification.pops_verify.reference_errors import reference_errors

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")

AX, AY, AZ = _exact.A
KX, KY, KZ = _exact.K
# AdaptiveCFL uses the per-axis max wave. For a=(1,1,1) the directional CFL
# is 3 times larger, so 0.15 keeps |a|_1 dt/dx = 0.45.
CFL = 0.15
MAX_STEPS = 100_000
T_END = float(_exact.T_END)
REQUIRED_NATIVE_DIM = int(_exact.REQUIRED_NATIVE_DIM)
RESOLUTIONS = tuple(_exact.RESOLUTIONS)
ORDER_THRESHOLD = 1.8


class NativeUnavailable(RuntimeError):
    """Raised when the Dim-3 native compile/run path cannot run."""


class _Authoring:
    __slots__ = ("case", "instance", "block", "frame", "n_cells", "program")

    def __init__(
        self,
        case: Any,
        instance: Any,
        block: Any,
        frame: Any,
        n_cells: int,
        program: Any,
    ) -> None:
        self.case = case
        self.instance = instance
        self.block = block
        self.frame = frame
        self.n_cells = n_cells
        self.program = program


def _cube_frame():
    return CartesianDomain(
        "tr01_unit_cube",
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ).frame(Cartesian3D())


def _sine_initial(frame):
    """Public analytic IC q0 + ε sin(2π k · x) at t = 0."""
    wave = 2.0 * np.pi
    profile = float(_exact.Q0) + float(_exact.EPS) * sin(
        wave
        * (
            float(KX) * analytic_x(frame)
            + float(KY) * analytic_y(frame)
            + float(KZ) * analytic_z(frame)
        )
    )
    return Analytic(frame=frame, components=(profile,))


def _author(n_cells: int) -> _Authoring:
    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    frame = _cube_frame()
    x_axis, y_axis, z_axis = frame.axes
    model = pops.Model("tr01_advection_sine_oblique_3d", frame=frame)
    state = model.state(
        "U",
        components=("q",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (q,) = state
    velocity = model.vector(
        "a",
        frame=frame,
        components={x_axis: AX, y_axis: AY, z_axis: AZ},
    )
    flux = model.flux(
        "advection_flux",
        frame=frame,
        state=state,
        components={
            x_axis: (AX * q,),
            y_axis: (AY * q,),
            z_axis: (AZ * q,),
        },
        waves={x_axis: (AX,), y_axis: (AY,), z_axis: (AZ,)},
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
    case = pops.Case("tr01_advection_sine_oblique_3d")
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
        case=case,
        instance=instance,
        block=tracer,
        frame=frame,
        n_cells=count,
        program=program,
    )


def build_case(n_cells: int = 16) -> pops.Case:
    """Author the 3-d periodic oblique Case. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = 16):
    """Validate and resolve the 3-d Case. Does not compile or call pops.run."""
    authored = _author(n_cells)
    layout = uniform_periodic_layout(
        authored.frame, (authored.n_cells, authored.n_cells, authored.n_cells)
    )
    return resolve_case(authored.case, layout=layout)


def _launched_native_dim() -> int | None:
    raw = os.environ.get("POPS_NATIVE_DIM")
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def _require_native_dim3() -> None:
    launched = _launched_native_dim()
    if launched != REQUIRED_NATIVE_DIM:
        raise NativeUnavailable(
            f"TR-01 requires POPS_NATIVE_DIM={REQUIRED_NATIVE_DIM} "
            f"(got {launched!r}); no 1-d/2-d fallback"
        )


def _confirm_selected_dim3() -> None:
    from pops._native_selector import selected_native_dimension

    selected = selected_native_dimension()
    if selected != REQUIRED_NATIVE_DIM:
        raise NativeUnavailable(
            f"selected native dimension is {selected!r}, not "
            f"{REQUIRED_NATIVE_DIM}; the loaded artifact is not Dim-3"
        )


def _native_unavailable_reason() -> str | None:
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def _unpack_field(field, n_cells: int) -> np.ndarray:
    array = np.ascontiguousarray(field, dtype=np.float64)
    count = int(n_cells)
    if array.shape == (count, count, count):
        return array
    if array.shape == (1, count, count, count):
        return array[0]
    flat = np.ravel(array)
    if flat.size != count**3:
        raise NativeUnavailable(
            f"TR-01 native field shape {array.shape} is not a {count}^3 cube"
        )
    return np.reshape(flat, (count, count, count))


def _oracle_cell_averages(n_cells: int, t: float) -> np.ndarray:
    lo, hi = _exact.cell_bounds(n_cells)

    def _u(x, y, z, time):
        return _exact.exact_sine_3d(x, y, z, time)

    return analytic_cell_averages(_u, lo, hi, t)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _live_provenance_digests() -> dict[str, str]:
    from pops.codegen.toolchain import pops_header_signature
    from pops.release import contract

    catalog = contract()["component_catalog_sha256"]
    header = pops_header_signature(repo_include())
    manifests: list[Path] = []
    explicit = os.environ.get("POPS_NATIVE_VARIANTS_ROOT")
    if explicit:
        candidate = Path(explicit) / "variants.json"
        if candidate.is_file():
            manifests.append(candidate)
    for item in getattr(pops, "__path__", ()):
        candidate = Path(item) / "_native" / "variants.json"
        if candidate.is_file():
            manifests.append(candidate)
    if not manifests:
        raise NativeUnavailable("no native variants.json for provenance digest")
    variant = _sha256_file(manifests[0])
    return {
        "component_catalog_digest": str(catalog),
        "native_header_signature": str(header),
        "native_variant_manifest_digest": variant,
    }


def _live_run_fields(n_cells: int, t_end: float) -> dict[str, Any]:
    cxx = default_cxx() or "unknown"
    return {
        "compiler": cxx,
        "build_type": os.environ.get("CMAKE_BUILD_TYPE", "Release"),
        "precision": "float64",
        "kokkos_execution_space": "OpenMP",
        "mpi_enabled": False,
        "mpi_library": "none",
        "mpi_thread_level_requested": "MPI_THREAD_SINGLE",
        "mpi_thread_level_provided": "MPI_THREAD_SINGLE",
        "hdf5_collective_enabled": False,
        "mpi_ranks": 1,
        "omp_threads_per_rank": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "gpus": 0,
        "resolution": [int(n_cells), int(n_cells), int(n_cells)],
        "block_size": [int(n_cells), int(n_cells), int(n_cells)],
        "amr_total_levels": 1,
        "refinement_ratio": 2,
        "subcycling": False,
        "time_program": "SSPRK2",
        "cfl": float(CFL),
        "final_time": float(t_end),
    }


def _write_run_provenance(output_dir: Path | None, n_cells: int, t_end: float) -> dict:
    document = collect_provenance(
        "TR-01",
        pops_native_dim=REQUIRED_NATIVE_DIM,
        dimension=REQUIRED_NATIVE_DIM,
        nodes=1,
        run=_live_run_fields(n_cells, t_end),
        doctor_ok=True,
        **_live_provenance_digests(),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_provenance(output_dir / f"provenance_n{int(n_cells)}.json", document)
        write_provenance(output_dir / "provenance.json", document)
    return document


def run_native(
    n_cells: int = 16,
    t_end: float = T_END,
    *,
    output_dir: Path | None = None,
):
    """Compile, bind, and run the 3-d Case on a Dim-3 native artifact."""
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    _require_native_dim3()
    authored = _author(n_cells)
    layout = uniform_periodic_layout(
        authored.frame, (authored.n_cells, authored.n_cells, authored.n_cells)
    )
    plan = resolve_case(authored.case, layout=layout)
    if getattr(plan, "resolved_dimension", None) != REQUIRED_NATIVE_DIM:
        raise NativeUnavailable(
            f"resolved dimension is {getattr(plan, 'resolved_dimension', None)!r}, "
            f"not {REQUIRED_NATIVE_DIM}"
        )
    artifact = pops.compile(plan)
    _confirm_selected_dim3()
    simulation = bind_public(artifact)
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = _unpack_field(
        simulation.state_global("tracer"), authored.n_cells
    )
    _write_run_provenance(output_dir, authored.n_cells, t_end)
    return field


def run_order_campaign(
    resolutions=RESOLUTIONS,
    *,
    t_end: float = T_END,
    output_dir: Path | None = None,
) -> dict:
    """Native 3-d Δx series: L1/L2/L∞ vs cell-averaged exact and observed orders."""
    steps = tuple(int(n) for n in resolutions)
    if len(steps) < 4:
        raise ValueError("TR-01 requires at least four resolutions")
    if any(n <= 0 for n in steps):
        raise ValueError("resolutions must be positive")
    linf = []
    l1 = []
    l2 = []
    fields = {}
    for n_cells in steps:
        field = run_native(n_cells, t_end=t_end, output_dir=output_dir)
        oracle = _oracle_cell_averages(n_cells, t_end)
        _, _, _, volumes = _exact.uniform_cell_mesh(n_cells)
        errors = reference_errors(field, oracle, volumes)
        linf.append(float(errors.linf))
        l1.append(float(errors.l1))
        l2.append(float(errors.l2))
        fields[n_cells] = field
    spacings = tuple(1.0 / float(n) for n in steps)
    orders = observed_order(linf, spacings)
    return {
        "case_id": "TR-01",
        "pops_native_dim": REQUIRED_NATIVE_DIM,
        "dimension": REQUIRED_NATIVE_DIM,
        "velocity": tuple(float(v) for v in _exact.A),
        "wave": tuple(float(v) for v in _exact.K),
        "t_end": float(t_end),
        "resolutions": steps,
        "spacings": spacings,
        "l1": tuple(l1),
        "l2": tuple(l2),
        "linf": tuple(linf),
        "orders": tuple(float(value) for value in orders),
        "threshold": ORDER_THRESHOLD,
        "fields": fields,
    }
