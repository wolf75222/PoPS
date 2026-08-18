"""TR-01 obligatory variants: 1-d / 2-d / 3-d sine, layouts, periods, blocks.

The catalog Case in ``run.py`` stays the 3-d canonical cube. This module
authors the plan §11 variants on the matching exact-rank native leaf.
``pops.run`` lives only in ``run_variant``.
"""
from __future__ import annotations

import json
import os
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
from verification.pops_verify.cell_averages import analytic_cell_averages
from verification.pops_verify.conservation import conservation_tolerance
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.interface_error import (
    band_max_abs_error,
    interface_band_mask,
    max_error_location,
)
from verification.pops_verify.phase import phase_error
from verification.pops_verify.provenance import collect_provenance, write_provenance
from verification.pops_verify.reference_errors import reference_errors

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
_exact = load_sibling_module(_CASE_DIR / "exact.py")

MAX_STEPS = 100_000
ORDER_THRESHOLD = 1.8
AMR_LAYOUTS = ("A-S0", "A-S2", "A-DP", "A-DT")
UNIFORM_LAYOUTS = ("U-C", "U-F")
LAYOUTS = UNIFORM_LAYOUTS + AMR_LAYOUTS


class NativeUnavailable(RuntimeError):
    """Raised when the matching exact-rank native path cannot run."""


class AuthoringPending(RuntimeError):
    """Raised when public validate/resolve cannot complete for a variant."""


class Variant:
    """One obligatory TR-01 run."""

    __slots__ = (
        "dim",
        "velocity",
        "wave",
        "n_cells",
        "periods",
        "layout",
        "block_size",
        "dt",
        "family",
    )

    def __init__(
        self,
        dim: int,
        velocity: tuple[float, ...],
        wave: tuple[float, ...],
        n_cells: int,
        periods: int = 1,
        layout: str = "U-C",
        block_size: int | None = None,
        dt: float | None = None,
        family: str = "spatial",
    ) -> None:
        if dim not in (1, 2, 3):
            raise ValueError("dim must be 1, 2, or 3")
        if len(velocity) != dim or len(wave) != dim:
            raise ValueError("velocity and wave must match dim")
        if n_cells <= 0 or periods <= 0:
            raise ValueError("n_cells and periods must be positive")
        if layout not in LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}")
        self.dim = int(dim)
        self.velocity = tuple(float(value) for value in velocity)
        self.wave = tuple(float(value) for value in wave)
        self.n_cells = int(n_cells)
        self.periods = int(periods)
        self.layout = str(layout)
        self.block_size = None if block_size is None else int(block_size)
        self.dt = None if dt is None else float(dt)
        self.family = str(family)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "velocity": list(self.velocity),
            "wave": list(self.wave),
            "n_cells": self.n_cells,
            "periods": self.periods,
            "layout": self.layout,
            "block_size": self.block_size,
            "dt": self.dt,
            "family": self.family,
            "vid": self.vid,
        }

    @property
    def t_end(self) -> float:
        return float(self.periods) * float(_exact.T_END)

    @property
    def mesh_cells(self) -> int:
        return int(self.n_cells) * (2 if self.layout == "U-F" else 1)

    @property
    def cfl(self) -> float | None:
        if self.dt is not None:
            return None
        return _exact.directional_cfl(self.velocity)

    @property
    def vid(self) -> str:
        vel = "x".join(f"{value:g}" for value in self.velocity)
        block = f"_b{self.block_size}" if self.block_size else ""
        time = f"_dt{self.dt:g}" if self.dt is not None else ""
        return (
            f"d{self.dim}_{self.family}_{self.layout}_n{self.n_cells}"
            f"_p{self.periods}_a{vel}{block}{time}"
        )


class _Authoring:
    __slots__ = (
        "case",
        "instance",
        "marker_instance",
        "block",
        "frame",
        "variant",
        "program",
    )

    def __init__(
        self,
        case: Any,
        instance: Any,
        marker_instance: Any,
        block: Any,
        frame: Any,
        variant: Variant,
        program: Any,
    ) -> None:
        self.case = case
        self.instance = instance
        self.marker_instance = marker_instance
        self.block = block
        self.frame = frame
        self.variant = variant
        self.program = program


def _frame(dim: int):
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D, Cartesian2D, Cartesian3D

    if dim == 1:
        return CartesianDomain("tr01_unit", (0.0,), (1.0,)).frame(Cartesian1D())
    if dim == 2:
        return CartesianDomain("tr01_unit", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    return CartesianDomain("tr01_unit", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)).frame(
        Cartesian3D()
    )


def _phase_expr(frame, wave):
    from pops.analytic import x as analytic_x, y as analytic_y, z as analytic_z

    coords = (analytic_x, analytic_y, analytic_z)
    phase = float(wave[0]) * coords[0](frame)
    for index in range(1, len(wave)):
        phase = phase + float(wave[index]) * coords[index](frame)
    return phase


def _sine_initial(frame, wave):
    from pops.analytic import sin
    from pops.lib.initial import Analytic

    profile = float(_exact.Q0) + float(_exact.EPS) * sin(
        2.0 * np.pi * _phase_expr(frame, wave)
    )
    return Analytic(frame=frame, components=(profile,))


def _static_marker(frame):
    from pops.analytic import where, x as analytic_x
    from pops.lib.initial import Analytic

    return Analytic(
        frame=frame,
        components=(where(analytic_x(frame) > 0.5, 1.0, 0.0),),
    )


def _prescribed_marker(frame, wave):
    from pops.analytic import abs as analytic_abs, sin, where
    from pops.lib.initial import Analytic

    crest = analytic_abs(sin(2.0 * np.pi * _phase_expr(frame, wave)))
    return Analytic(frame=frame, components=(where(crest > 0.5, 1.0, 0.0),))


def _scalar_transport(name, frame, velocity):
    import pops
    from pops.math import ddt, div
    from pops.representations import Conservative
    from pops.spaces import CellState

    model = pops.Model(name, frame=frame)
    state = model.state(
        "U",
        components=("q",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (q,) = state
    axes = frame.axes
    components = {axis: float(speed) for axis, speed in zip(axes, velocity, strict=True)}
    vector = model.vector("a", frame=frame, components=components)
    flux = model.flux(
        f"{name}_flux",
        frame=frame,
        state=state,
        components={axis: (float(speed) * q,) for axis, speed in zip(axes, velocity)},
        waves={axis: (float(speed),) for axis, speed in zip(axes, velocity)},
    )
    rate = model.rate(f"{name}_rate", equation=ddt(state) == -div(flux))
    return model, state, flux, rate, vector


def _add_fv(case, block, state, flux, rate, velocity):
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume

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
    case.numerics(numerics, block=block)


def author(variant: Variant) -> _Authoring:
    """Author one variant Case. Does not compile or run."""
    import pops
    from pops.initial import InitialCondition
    from pops.lib.time import RungeKutta, RungeKuttaRoute, SSPRK2, SSPRK2_TABLEAU
    from pops.projection import ConservativeCellAverage
    from pops.time import AdaptiveCFL, FixedDt

    frame = _frame(variant.dim)
    tracer_model, tracer_state, tracer_flux, tracer_rate, tracer_velocity = (
        _scalar_transport(f"tr01_{variant.vid}_q", frame, variant.velocity)
    )
    case = pops.Case(f"tr01_{variant.vid}")
    tracer = case.block("tracer", model=tracer_model, states=(tracer_state,))
    instance = tracer[tracer_state]
    _add_fv(case, tracer, tracer_state, tracer_flux, tracer_rate, tracer_velocity)

    marker_instance = None
    needs_marker = variant.layout in ("A-S0", "A-S2", "A-DP")
    if needs_marker:
        marker_speed = (
            variant.velocity if variant.layout == "A-DP" else (0.0,) * variant.dim
        )
        marker_model, marker_state, marker_flux, marker_rate, marker_velocity = (
            _scalar_transport(f"tr01_{variant.vid}_m", frame, marker_speed)
        )
        marker = case.block("marker", model=marker_model, states=(marker_state,))
        marker_instance = marker[marker_state]
        _add_fv(case, marker, marker_state, marker_flux, marker_rate, marker_velocity)
        program = RungeKutta(
            routes=(
                RungeKuttaRoute(instance, tracer_rate),
                RungeKuttaRoute(marker_instance, marker_rate),
            ),
            tableau=SSPRK2_TABLEAU,
        )
    else:
        program = SSPRK2(instance, rate=tracer_rate)

    if variant.dt is not None:
        program.step_strategy(FixedDt(dt=float(variant.dt)))
    else:
        program.step_strategy(AdaptiveCFL(cfl=float(variant.cfl)))
    case.program(program)
    attach_case_diagnostics(case, tracer, program, every_n=1_000_000)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=_sine_initial(frame, variant.wave),
            projection=ConservativeCellAverage(),
        )
    )
    if marker_instance is not None:
        marker_value = (
            _prescribed_marker(frame, variant.wave)
            if variant.layout == "A-DP"
            else _static_marker(frame)
        )
        case.initials.add(
            InitialCondition(
                state=marker_instance,
                value=marker_value,
                projection=ConservativeCellAverage(),
            )
        )
    return _Authoring(
        case=case,
        instance=instance,
        marker_instance=marker_instance,
        block=tracer,
        frame=frame,
        variant=variant,
        program=program,
    )


def _cells(variant: Variant) -> tuple[int, ...]:
    count = variant.mesh_cells
    return (count,) * variant.dim


def layout_for(authored: _Authoring):
    """Return the public Uniform or AMR layout for the authored variant."""
    variant = authored.variant
    if variant.layout in UNIFORM_LAYOUTS:
        return uniform_periodic_layout(authored.frame, _cells(variant))

    from pops.amr import (
        AMRClockRelation,
        AMRExecution,
        AMRHierarchy,
        AMRRegrid,
        AMRTagging,
        AMRTransfer,
        Buffer,
        Coarsen,
        ConflictPolicy,
        EqualityPolicy,
        Hysteresis,
        PatchLayout,
        Tag,
    )
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.math import ValueExpr
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.params import RuntimeParam
    from pops.time import every

    case = authored.case
    transfer = AMRTransfer()
    transfer.state(authored.instance, StateTransfer())
    if authored.marker_instance is not None:
        transfer.state(authored.marker_instance, StateTransfer())

    if variant.layout == "A-DT":
        refine = case.param(
            RuntimeParam("tr01_refine_q", default=float(_exact.Q0) + 0.3 * _exact.EPS)
        )
        coarsen = case.param(
            RuntimeParam("tr01_coarsen_q", default=float(_exact.Q0) + 0.15 * _exact.EPS)
        )
        rules = (
            Tag(ValueExpr(authored.instance) > case.value(refine)),
            Coarsen(ValueExpr(authored.instance) < case.value(coarsen)),
            Buffer(cells=2),
        )
        regrid = AMRRegrid(schedule=every(2, clock=authored.program.clock))
    else:
        threshold = case.param(RuntimeParam("tr01_refine_marker", default=0.5))
        rules = (
            Tag(ValueExpr(authored.marker_instance) > case.value(threshold)),
            Buffer(cells=1),
        )
        regrid = (
            AMRRegrid(schedule=every(2, clock=authored.program.clock))
            if variant.layout == "A-DP"
            else AMRRegrid.frozen()
        )

    if variant.layout == "A-S2":
        execution = AMRExecution.subcycled((AMRClockRelation(0, 1, 2),))
    else:
        execution = AMRExecution.synchronous()

    patch = PatchLayout()
    if variant.block_size is not None:
        patch = PatchLayout(
            distribute_coarse=True,
            coarse_max_grid=int(variant.block_size),
        )
    return AMR(
        grid=CartesianGrid(
            frame=authored.frame,
            cells=_cells(variant),
            periodic=PeriodicAxes(authored.frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=rules,
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=regrid,
        transfer=transfer,
        execution=execution,
        patch_layout=patch,
    )


def build_case(variant: Variant):
    """Author the variant Case. Does not compile or run."""
    return author(variant).case


def resolve_plan(variant: Variant):
    """Validate and resolve the variant. Does not compile or call pops.run."""
    try:
        authored = author(variant)
        return resolve_case(authored.case, layout=layout_for(authored))
    except Exception as exc:
        raise AuthoringPending(
            f"{variant.vid} resolve failed: {type(exc).__name__}: {exc}"
        ) from exc


def _launched_native_dim() -> int | None:
    raw = os.environ.get("POPS_NATIVE_DIM")
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def _require_native_dim(dim: int) -> None:
    launched = _launched_native_dim()
    if launched != int(dim):
        raise NativeUnavailable(
            f"variant requires POPS_NATIVE_DIM={dim} (got {launched!r})"
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


def _unpack_field(field, n_cells: int, dim: int) -> np.ndarray:
    array = np.ascontiguousarray(field, dtype=np.float64)
    count = int(n_cells)
    rank = int(dim)
    shape = (count,) * rank
    if array.shape == shape:
        return array
    if array.shape == (1,) + shape:
        return array[0]
    flat = np.ravel(array)
    if flat.size != count**rank:
        raise NativeUnavailable(
            f"native field shape {array.shape} is not {count}^{rank}"
        )
    return np.reshape(flat, shape)


def _oracle(variant: Variant, t: float) -> np.ndarray:
    lo, hi = _exact.cell_bounds_nd(variant.mesh_cells, variant.dim)

    def _u(*args):
        *coords, time = args
        return _exact.exact_sine_nd(coords, time, a=variant.velocity, k=variant.wave)

    return analytic_cell_averages(_u, lo, hi, t)


def _interface_mask(variant: Variant, coords) -> np.ndarray | None:
    if variant.layout not in AMR_LAYOUTS:
        return None
    xx = coords[0]
    width = 1.0 / float(variant.mesh_cells)
    return interface_band_mask(np.abs(xx - 0.5), h_fine=0.5 * width, band_cells=4)


def _block_face_ratio(abs_error: np.ndarray, block_size: int | None) -> float | None:
    if block_size is None or block_size <= 0:
        return None
    face = np.zeros(abs_error.shape, dtype=bool)
    for axis in range(abs_error.ndim):
        slicer = [slice(None)] * abs_error.ndim
        slicer[axis] = slice(0, None, int(block_size))
        face[tuple(slicer)] = True
        slicer[axis] = slice(int(block_size) - 1, None, int(block_size))
        face[tuple(slicer)] = True
    interior = ~face
    if not np.any(face) or not np.any(interior):
        return None
    bulk = float(np.max(abs_error[interior]))
    if bulk == 0.0:
        return None
    return float(np.max(abs_error[face]) / bulk)


def field_diagnostics(field, oracle, volumes, *, variant: Variant) -> dict[str, Any]:
    """Norms, phase, amplitude, mass, spectrum, argmax, optional E_cf."""
    errors = reference_errors(field, oracle, volumes)
    residual = np.asarray(field, dtype=np.float64) - float(_exact.Q0)
    reference = np.asarray(oracle, dtype=np.float64) - float(_exact.Q0)
    amp_num = 0.5 * (float(np.max(field)) - float(np.min(field)))
    amplitude_loss = 1.0 - amp_num / float(_exact.EPS)
    try:
        phase = float(phase_error(residual.ravel(), reference.ravel()))
    except ValueError:
        phase = None
    integral = float(np.sum(np.asarray(field) * np.asarray(volumes)))
    volume = float(np.sum(volumes))
    mass_error = integral - float(_exact.Q0) * volume
    abs_error = np.abs(np.asarray(field) - np.asarray(oracle))
    spectrum = np.abs(np.fft.fftn(abs_error))
    spectrum.ravel()[0] = 0.0
    peak = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
    coords, _ = _exact.uniform_cell_mesh_nd(variant.mesh_cells, variant.dim)
    interface = _interface_mask(variant, coords)
    e_cf = e_bulk = None
    if interface is not None and np.any(interface) and np.any(~interface):
        e_cf = band_max_abs_error(field, oracle, interface)
        e_bulk = band_max_abs_error(field, oracle, ~interface)
        location = max_error_location(field, oracle, np.ones(field.shape, dtype=bool))
    else:
        location = tuple(int(i) for i in np.unravel_index(int(np.argmax(abs_error)), abs_error.shape))
    return {
        "l1": float(errors.l1),
        "l2": float(errors.l2),
        "linf": float(errors.linf),
        "phase_error": phase,
        "amplitude_loss": float(amplitude_loss),
        "integral": integral,
        "mass_error": float(mass_error),
        "error_spectrum_peak": [int(i) for i in peak],
        "max_error_index": [int(i) for i in location],
        "e_cf": None if e_cf is None else float(e_cf),
        "e_bulk": None if e_bulk is None else float(e_bulk),
        "block_face_ratio": _block_face_ratio(abs_error, variant.block_size),
    }


def _gather_field(simulation, variant: Variant) -> np.ndarray:
    count = variant.mesh_cells
    if variant.layout in AMR_LAYOUTS:
        raw = simulation.block_level_state_global("tracer", 0)
        return _unpack_field(raw, variant.n_cells, variant.dim)
    return _unpack_field(simulation.state_global("tracer"), count, variant.dim)


def _write_provenance(variant: Variant, output_dir: Path | None) -> None:
    from tests.python.support.requirements import default_cxx, repo_include
    from pops.codegen.toolchain import pops_header_signature
    from pops.release import contract
    import pops

    if output_dir is None:
        return
    catalog = contract()["component_catalog_sha256"]
    header = pops_header_signature(repo_include())
    manifests = []
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
        return
    import hashlib

    digest = hashlib.sha256(manifests[0].read_bytes()).hexdigest()
    cells = variant.mesh_cells
    document = collect_provenance(
        "TR-01",
        pops_native_dim=variant.dim,
        dimension=variant.dim,
        nodes=1,
        run={
            "compiler": default_cxx() or "unknown",
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
            "resolution": [cells] * variant.dim,
            "block_size": [int(variant.block_size or cells)] * variant.dim,
            "amr_total_levels": 2 if variant.layout in AMR_LAYOUTS else 1,
            "refinement_ratio": 2,
            "subcycling": variant.layout == "A-S2",
            "time_program": "SSPRK2",
            "cfl": float(
                variant.cfl
                if variant.cfl is not None
                else variant.dt * float(variant.mesh_cells)
            ),
            "final_time": variant.t_end,
        },
        doctor_ok=True,
        component_catalog_digest=str(catalog),
        native_header_signature=str(header),
        native_variant_manifest_digest=digest,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(output_dir / f"provenance_{variant.vid}.json", document)


def run_variant(variant: Variant, *, output_dir: Path | None = None) -> dict[str, Any]:
    """Compile, bind, and run one variant. Returns diagnostics or a failure row."""
    import pops

    row = {
        "id": variant.vid,
        "dim": variant.dim,
        "velocity": list(variant.velocity),
        "wave": list(variant.wave),
        "n_cells": variant.n_cells,
        "mesh_cells": variant.mesh_cells,
        "periods": variant.periods,
        "layout": variant.layout,
        "block_size": variant.block_size,
        "dt": variant.dt,
        "family": variant.family,
        "t_end": variant.t_end,
        "cfl": variant.cfl,
        "status": "failed",
    }
    missing = _native_unavailable_reason()
    if missing:
        row["status"] = "unavailable"
        row["error"] = missing
        return row
    try:
        _require_native_dim(variant.dim)
        authored = author(variant)
        plan = resolve_case(authored.case, layout=layout_for(authored))
        if getattr(plan, "resolved_dimension", None) != variant.dim:
            raise NativeUnavailable(
                f"resolved dimension is {getattr(plan, 'resolved_dimension', None)!r}"
            )
        artifact = pops.compile(plan)
        from pops._native_selector import selected_native_dimension

        if selected_native_dimension() != variant.dim:
            raise NativeUnavailable("loaded native leaf is not the requested dim")
        simulation = bind_public(artifact)
        pops.run(simulation, t_end=variant.t_end, max_steps=MAX_STEPS)
        field = _gather_field(simulation, variant)
        oracle_cells = (
            variant.n_cells if variant.layout in AMR_LAYOUTS else variant.mesh_cells
        )
        measure = Variant(
            dim=variant.dim,
            velocity=variant.velocity,
            wave=variant.wave,
            n_cells=oracle_cells,
            periods=variant.periods,
            layout=variant.layout,
            block_size=variant.block_size,
            family=variant.family,
        )
        oracle = _oracle(
            Variant(
                dim=variant.dim,
                velocity=variant.velocity,
                wave=variant.wave,
                n_cells=oracle_cells,
                periods=variant.periods,
                layout="U-C",
                family=variant.family,
            ),
            variant.t_end,
        )
        _, volumes = _exact.uniform_cell_mesh_nd(oracle_cells, variant.dim)
        diagnostics = field_diagnostics(field, oracle, volumes, variant=measure)
        n_steps = int(getattr(simulation, "n_accepted_steps", 0) or 0)
        tol = conservation_tolerance(
            1.0, abs_tol=1.0e-12, rel_tol=1.0e-12, n_updates=max(n_steps, 1), c=100.0
        )
        diagnostics["conservation_tolerance"] = float(tol)
        diagnostics["conservation_ok"] = abs(diagnostics["mass_error"]) <= tol
        row.update(diagnostics)
        row["finite"] = bool(np.isfinite(field).all())
        exploded = (not row["finite"]) or float(diagnostics["linf"]) > 1.0
        row["status"] = "failed" if exploded else "ok"
        if exploded:
            row["error"] = "solution exploded or is non-finite"
        try:
            _write_provenance(variant, output_dir)
        except Exception as exc:
            row["provenance_error"] = f"{type(exc).__name__}: {exc}"
    except AuthoringPending as exc:
        row["status"] = "unsupported"
        row["error"] = str(exc)
    except NativeUnavailable as exc:
        row["status"] = "unavailable"
        row["error"] = str(exc)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _add(rows: list[Variant], **kwargs) -> None:
    rows.append(Variant(**kwargs))


def catalog(*, dim: int | None = None, smoke: bool = False) -> tuple[Variant, ...]:
    """Obligatory TR-01 variants, optionally filtered by native dimension."""
    rows: list[Variant] = []
    if smoke:
        if dim in (None, 1):
            _add(rows, dim=1, velocity=(1.0,), wave=(1.0,), n_cells=16, family="spatial")
        if dim in (None, 2):
            _add(
                rows,
                dim=2,
                velocity=(1.0, 0.0),
                wave=(1.0, 2.0),
                n_cells=16,
                family="spatial",
            )
        if dim in (None, 3):
            _add(
                rows,
                dim=3,
                velocity=(1.0, 0.0, 0.0),
                wave=(1.0, 2.0, 3.0),
                n_cells=16,
                family="spatial",
            )
        return tuple(row for row in rows if dim is None or row.dim == dim)

    if dim in (None, 1):
        wave1 = (1.0,)
        for speed in (1.0, -1.0):
            for n_cells in (16, 32, 64, 128, 256):
                _add(
                    rows,
                    dim=1,
                    velocity=(speed,),
                    wave=wave1,
                    n_cells=n_cells,
                    family="spatial",
                )
        for periods in (2, 4):
            _add(
                rows,
                dim=1,
                velocity=(1.0,),
                wave=wave1,
                n_cells=64,
                periods=periods,
                family="periods",
            )
        for n_cells in (32, 64):
            _add(
                rows,
                dim=1,
                velocity=(1.0,),
                wave=wave1,
                n_cells=n_cells,
                layout="U-F",
                family="uniform_fine",
            )
        for layout in AMR_LAYOUTS:
            _add(
                rows,
                dim=1,
                velocity=(1.0,),
                wave=wave1,
                n_cells=32,
                layout=layout,
                family="amr",
            )
        for block in (8, 16, 32, 64):
            _add(
                rows,
                dim=1,
                velocity=(1.0,),
                wave=wave1,
                n_cells=64,
                layout="A-S0",
                block_size=block,
                family="blocks",
            )
        for step in (1.0 / 64.0, 1.0 / 128.0, 1.0 / 256.0, 1.0 / 512.0):
            _add(
                rows,
                dim=1,
                velocity=(1.0,),
                wave=wave1,
                n_cells=128,
                dt=step,
                family="temporal",
            )

    if dim in (None, 2):
        wave2 = (1.0, 2.0)
        for velocity in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.37)):
            for n_cells in (16, 32, 64, 128):
                _add(
                    rows,
                    dim=2,
                    velocity=velocity,
                    wave=wave2,
                    n_cells=n_cells,
                    family="spatial",
                )
        for periods in (2, 4):
            _add(
                rows,
                dim=2,
                velocity=(1.0, 1.0),
                wave=wave2,
                n_cells=32,
                periods=periods,
                family="periods",
            )
        _add(
            rows,
            dim=2,
            velocity=(1.0, 1.0),
            wave=wave2,
            n_cells=32,
            layout="U-F",
            family="uniform_fine",
        )
        for layout in AMR_LAYOUTS:
            _add(
                rows,
                dim=2,
                velocity=(1.0, 1.0),
                wave=wave2,
                n_cells=16,
                layout=layout,
                family="amr",
            )
        for block in (8, 16, 32):
            _add(
                rows,
                dim=2,
                velocity=(1.0, 1.0),
                wave=wave2,
                n_cells=32,
                layout="A-S0",
                block_size=block,
                family="blocks",
            )
        _add(
            rows,
            dim=2,
            velocity=(0.37, 1.0),
            wave=(2.0, 1.0),
            n_cells=32,
            family="perm",
        )
        for step in (1.0 / 64.0, 1.0 / 128.0, 1.0 / 256.0):
            _add(
                rows,
                dim=2,
                velocity=(1.0, 0.0),
                wave=wave2,
                n_cells=64,
                dt=step,
                family="temporal",
            )

    if dim in (None, 3):
        wave3 = (1.0, 2.0, 3.0)
        for velocity in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            for n_cells in (16, 32, 64):
                _add(
                    rows,
                    dim=3,
                    velocity=velocity,
                    wave=wave3,
                    n_cells=n_cells,
                    family="spatial",
                )
        for n_cells in (16, 32, 64):
            _add(
                rows,
                dim=3,
                velocity=(1.0, 1.0, 1.0),
                wave=wave3,
                n_cells=n_cells,
                family="spatial",
            )
            _add(
                rows,
                dim=3,
                velocity=(1.0, 0.37, 0.61),
                wave=wave3,
                n_cells=n_cells,
                family="spatial",
            )
        _add(
            rows,
            dim=3,
            velocity=(1.0, 0.0, 0.0),
            wave=wave3,
            n_cells=256,
            family="spatial",
        )
        for periods in (2, 4):
            _add(
                rows,
                dim=3,
                velocity=(1.0, 1.0, 1.0),
                wave=wave3,
                n_cells=16,
                periods=periods,
                family="periods",
            )
        _add(
            rows,
            dim=3,
            velocity=(1.0, 1.0, 1.0),
            wave=wave3,
            n_cells=16,
            layout="U-F",
            family="uniform_fine",
        )
        _add(
            rows,
            dim=3,
            velocity=(1.0, 1.0, 1.0),
            wave=wave3,
            n_cells=16,
            layout="A-S0",
            family="amr",
        )
        _add(
            rows,
            dim=3,
            velocity=(0.37, 0.61, 1.0),
            wave=(2.0, 3.0, 1.0),
            n_cells=16,
            family="perm",
        )
        for step in (1.0 / 32.0, 1.0 / 64.0, 1.0 / 128.0):
            _add(
                rows,
                dim=3,
                velocity=(1.0, 0.0, 0.0),
                wave=wave3,
                n_cells=32,
                dt=step,
                family="temporal",
            )
    return tuple(row for row in rows if dim is None or row.dim == dim)


def _series_orders(
    results: list[dict], *, family: str, kind: str, norm: str = "linf"
) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in results:
        usable = row.get("status") == "ok" or (
            row.get(norm) is not None and float(row.get(norm) or 1.0e300) < 1.0
        )
        if not usable or row.get("family") != family:
            continue
        if row.get(norm) is None:
            continue
        if family == "temporal":
            key = (
                row["dim"],
                tuple(row["velocity"]),
                row["n_cells"],
                row["layout"],
                row["periods"],
            )
            sort_key = row["dt"]
        else:
            key = (
                row["dim"],
                tuple(row["velocity"]),
                row["layout"],
                row["periods"],
                row.get("block_size"),
            )
            sort_key = row["mesh_cells"]
        groups.setdefault(key, []).append((sort_key, row))
    orders = []
    for key, items in groups.items():
        items.sort(key=lambda pair: pair[0])
        if len(items) < 2:
            continue
        spacings = []
        errors = []
        for sort_key, row in items:
            if family == "temporal":
                spacings.append(float(row["dt"]))
            else:
                spacings.append(1.0 / float(row["mesh_cells"]))
            errors.append(float(row[norm]))
        try:
            observed = observed_order(errors, spacings)
        except ValueError:
            continue
        for value in observed:
            orders.append(
                {
                    "case_id": "TR-01",
                    "kind": kind,
                    "variable": f"q_{norm}",
                    "group": [key[0], list(key[1])],
                    "observed_order": float(value),
                    "threshold": ORDER_THRESHOLD,
                    "norm": norm,
                }
            )
    return orders


def summarize(results: list[dict]) -> dict[str, Any]:
    """Aggregate orders, conservation, and acceptance against the plan."""
    spatial = _series_orders(results, family="spatial", kind="spatial", norm="linf")
    spatial_l1 = _series_orders(results, family="spatial", kind="spatial", norm="l1")
    temporal = _series_orders(results, family="temporal", kind="temporal", norm="linf")
    global_orders = _series_orders(results, family="spatial", kind="global", norm="linf")
    ok = [row for row in results if row.get("status") == "ok"]
    failed = [row for row in results if row.get("status") == "failed"]
    unsupported = [row for row in results if row.get("status") == "unsupported"]
    unavailable = [row for row in results if row.get("status") == "unavailable"]
    passing_spatial = [
        row for row in spatial if row["observed_order"] >= ORDER_THRESHOLD
    ]
    passing_l1 = [
        row for row in spatial_l1 if row["observed_order"] >= ORDER_THRESHOLD
    ]
    conservation_ok = all(row.get("conservation_ok") for row in ok) if ok else False
    block_peaks = [
        row
        for row in ok
        if row.get("block_face_ratio") is not None and row["block_face_ratio"] > 4.0
    ]
    amr_ok = [row for row in ok if row["layout"] in AMR_LAYOUTS]
    return {
        "case_id": "TR-01",
        "threshold": ORDER_THRESHOLD,
        "n_planned": len(results),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "n_unsupported": len(unsupported),
        "n_unavailable": len(unavailable),
        "spatial_orders": spatial,
        "spatial_orders_l1": spatial_l1,
        "temporal_orders": temporal,
        "global_orders": global_orders,
        "spatial_pairs_ge_1_8": len(passing_spatial),
        "spatial_pairs": len(spatial),
        "spatial_l1_pairs_ge_1_8": len(passing_l1),
        "spatial_l1_pairs": len(spatial_l1),
        "acceptance_order_met": bool(spatial) and len(passing_spatial) == len(spatial),
        "acceptance_l1_order_met": bool(spatial_l1) and len(passing_l1) == len(spatial_l1),
        "conservation_ok": conservation_ok,
        "fixed_block_peaks": [row["id"] for row in block_peaks],
        "amr_ran": len(amr_ok),
        "layouts_ok": sorted({row["layout"] for row in ok}),
        "dims_ok": sorted({row["dim"] for row in ok}),
        "failed": [row["id"] for row in failed],
        "unsupported": [row["id"] for row in unsupported],
        "unavailable": [row["id"] for row in unavailable],
    }


def run_campaign(
    *,
    dim: int,
    output_dir: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the catalog for one native dimension and write JSON evidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "results.jsonl"
    done: dict[str, dict] = {}
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            existing = json.loads(line)
            done[str(existing.get("id"))] = existing
    variants = catalog(dim=dim, smoke=smoke)
    if os.environ.get("TR01_SKIP_HEAVY") == "1":
        variants = tuple(row for row in variants if row.n_cells < 256)
    results = list(done.values())
    for variant in variants:
        if variant.vid in done:
            print(f"{variant.vid} skip-existing {done[variant.vid].get('status')}", flush=True)
            continue
        row = run_variant(variant, output_dir=output_dir)
        results.append(row)
        (output_dir / "results.jsonl").open("a", encoding="utf-8").write(
            json.dumps(row) + "\n"
        )
        status = row.get("status")
        detail = row.get("linf", row.get("error", ""))
        print(f"{variant.vid} {status} {detail}", flush=True)
    summary = summarize(results)
    payload = {
        "schema": "pops.verification.tr01_complement.v1",
        "dim": dim,
        "smoke": smoke,
        "variants": [variant.to_dict() for variant in variants],
        "results": results,
        "summary": summary,
    }
    (output_dir / "complement.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return payload
