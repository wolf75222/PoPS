"""TR-01 public sine advection: 1-d/2-d restrictions and canonical 3-d.

Canonical 3-d (§9.2): a = (1, 1, 1), k = (1, 2, 3), T = 1 on [0, 1]^3.
The order campaign uses public WENO5-Z. VanLeer is a separately labeled TVD
variant and is not the order-2 acceptance proof. Constant-CFL series are
global, never isolated spatial. ``pops.run`` lives in ``run_native``,
``run_order_campaign``, and ``run_temporal_campaign`` only.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pops
from pops.analytic import sin, where, x as analytic_x, y as analytic_y, z as analytic_z
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D, Cartesian2D, Cartesian3D
from pops.initial import InitialCondition
from pops.lib.initial import Analytic
from pops.lib.time import RK4, SSPRK2, SSPRK3
from pops.math import ddt, div
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL, FixedDt
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
from verification.pops_verify.conservation import conservation_tolerance
from verification.pops_verify.native_evidence import program_bytes_from_artifact
from verification.pops_verify.provenance import collect_provenance, write_provenance
from verification.pops_verify.reference_errors import reference_errors

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")

AX, AY, AZ = _exact.A
KX, KY, KZ = _exact.K
MAX_STEPS = 100_000
T_END = float(_exact.T_END)
REQUIRED_NATIVE_DIM = int(_exact.REQUIRED_NATIVE_DIM)
RESOLUTIONS = tuple(_exact.RESOLUTIONS)
ORDER_THRESHOLD = 1.8
AMR_LAYOUTS = ("A-S0", "A-S2", "A-DP", "A-DT")
UNIFORM_LAYOUTS = ("U-C", "U-F")
LAYOUTS = UNIFORM_LAYOUTS + AMR_LAYOUTS


class NativeUnavailable(RuntimeError):
    """Raised when the requested exact-rank native path cannot run."""


class AuthoringPending(RuntimeError):
    """Raised when public validate/resolve cannot complete for a variant."""


class Tr01Config:
    """One executable TR-01 configuration under the single case authority."""

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
        "label",
        "reconstruction",
        "time_program",
        "dt_scaling",
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
        family: str = "global",
        label: str = "variant",
        reconstruction: str = "weno5z",
        time_program: str = "SSPRK2",
        dt_scaling: str = "cfl",
    ) -> None:
        if dim not in (1, 2, 3):
            raise ValueError("dim must be 1, 2, or 3")
        if len(velocity) != dim or len(wave) != dim:
            raise ValueError("velocity and wave must match dim")
        if n_cells <= 0 or periods <= 0:
            raise ValueError("n_cells and periods must be positive")
        if layout not in LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}")
        if reconstruction not in {"weno5z", "vanleer"}:
            raise ValueError(f"unknown reconstruction {reconstruction!r}")
        if time_program not in {"SSPRK2", "SSPRK3", "RK4"}:
            raise ValueError(f"unknown time_program {time_program!r}")
        if dt_scaling not in {"cfl", "h2", "fixed"}:
            raise ValueError(f"unknown dt_scaling {dt_scaling!r}")
        self.dim = int(dim)
        self.velocity = tuple(float(value) for value in velocity)
        self.wave = tuple(float(value) for value in wave)
        self.n_cells = int(n_cells)
        self.periods = int(periods)
        self.layout = str(layout)
        self.block_size = None if block_size is None else int(block_size)
        self.dt = None if dt is None else float(dt)
        self.family = str(family)
        self.label = str(label)
        self.reconstruction = str(reconstruction)
        self.time_program = str(time_program)
        self.dt_scaling = str(dt_scaling)

    def replace(self, **kwargs) -> Tr01Config:
        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(kwargs)
        return Tr01Config(**values)

    @property
    def t_end(self) -> float:
        return float(self.periods) * float(_exact.T_END)

    @property
    def mesh_cells(self) -> int:
        return int(self.n_cells) * (2 if self.layout == "U-F" else 1)

    @property
    def cfl(self) -> float | None:
        if self.dt is not None or self.dt_scaling != "cfl":
            return None
        return _exact.directional_cfl(self.velocity)

    @property
    def step_dt(self) -> float | None:
        if self.dt is not None:
            return float(self.dt)
        if self.dt_scaling == "h2" or self.family == "spatial":
            width = 1.0 / float(self.mesh_cells)
            return 4.0 * width * width
        return None

    @property
    def config_id(self) -> str:
        if self.label in {"canonical", "restriction_1d", "restriction_2d"}:
            return self.label
        vel = "x".join(f"{value:g}" for value in self.velocity)
        block = f"_b{self.block_size}" if self.block_size else ""
        time = f"_dt{self.dt:g}" if self.dt is not None else ""
        scheme = "" if self.reconstruction == "weno5z" else f"_{self.reconstruction}"
        return (
            f"d{self.dim}_{self.family}_{self.layout}_n{self.n_cells}"
            f"_p{self.periods}_a{vel}{block}{time}{scheme}"
        )


class _Authoring:
    __slots__ = (
        "case",
        "instance",
        "marker_instance",
        "block",
        "frame",
        "config",
        "program",
        "n_cells",
    )

    def __init__(
        self,
        case: Any,
        instance: Any,
        block: Any,
        frame: Any,
        config: Tr01Config,
        program: Any,
        marker_instance: Any = None,
    ) -> None:
        self.case = case
        self.instance = instance
        self.marker_instance = marker_instance
        self.block = block
        self.frame = frame
        self.config = config
        self.program = program
        self.n_cells = config.n_cells


def _frame(dim: int):
    if dim == 1:
        return CartesianDomain("tr01_unit", (0.0,), (1.0,)).frame(Cartesian1D())
    if dim == 2:
        return CartesianDomain("tr01_unit", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    return CartesianDomain(
        "tr01_unit_cube", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    ).frame(Cartesian3D())


def _phase_expr(frame, wave):
    coords = (analytic_x, analytic_y, analytic_z)
    phase = float(wave[0]) * coords[0](frame)
    for index in range(1, len(wave)):
        phase = phase + float(wave[index]) * coords[index](frame)
    return phase


def _sine_initial(frame, wave):
    profile = float(_exact.Q0) + float(_exact.EPS) * sin(
        2.0 * np.pi * _phase_expr(frame, wave)
    )
    return Analytic(frame=frame, components=(profile,))


def _half_marker(frame):
    """Static spatial indicator: 1 on the first-axis half-domain."""
    return Analytic(
        frame=frame,
        components=(where(analytic_x(frame) > 0.5, 1.0, 0.0),),
    )


def _reconstruction_brick(name: str):
    if name == "vanleer":
        return reconstruction.MUSCL(limiters.VanLeer())
    return reconstruction.WENO5Z()


def _make_program(name: str, instance, rate):
    if name == "RK4":
        return RK4(instance, rate=rate)
    if name == "SSPRK3":
        return SSPRK3(instance, rate=rate)
    return SSPRK2(instance, rate=rate)


def _scalar_transport(name, frame, velocity):
    model = pops.Model(name, frame=frame)
    state = model.state(
        "U",
        components=("q",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (q,) = state
    axes = frame.axes
    vector = model.vector(
        "a",
        frame=frame,
        components={axis: float(speed) for axis, speed in zip(axes, velocity, strict=True)},
    )
    flux = model.flux(
        f"{name}_flux",
        frame=frame,
        state=state,
        components={
            axis: (float(speed) * q,) for axis, speed in zip(axes, velocity, strict=True)
        },
        waves={axis: (float(speed),) for axis, speed in zip(axes, velocity, strict=True)},
    )
    rate = model.rate(f"{name}_rate", equation=ddt(state) == -div(flux))
    return model, state, flux, rate, vector


def _add_fv(case, block, state, flux, rate, velocity, reconstruction_name: str):
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=_reconstruction_brick(reconstruction_name),
            riemann=riemann.ScalarUpwind(velocity=velocity),
        ),
    )
    case.numerics(numerics, block=block)


def author(config: Tr01Config) -> _Authoring:
    """Author one configuration. Does not compile or run."""
    frame = _frame(config.dim)
    model, state, flux, rate, velocity = _scalar_transport(
        f"tr01_{config.config_id}_q", frame, config.velocity
    )
    case = pops.Case(f"tr01_{config.config_id}")
    tracer = case.block("tracer", model=model, states=(state,))
    instance = tracer[state]
    _add_fv(case, tracer, state, flux, rate, velocity, config.reconstruction)
    marker_instance = None
    if config.layout in {"A-S0", "A-S2"}:
        zeros = tuple(0.0 for _ in config.velocity)
        marker_model, marker_state, marker_flux, marker_rate, marker_velocity = (
            _scalar_transport(f"tr01_{config.config_id}_m", frame, zeros)
        )
        marker = case.block("marker", model=marker_model, states=(marker_state,))
        marker_instance = marker[marker_state]
        _add_fv(
            case,
            marker,
            marker_state,
            marker_flux,
            marker_rate,
            marker_velocity,
            "vanleer",
        )
    program = _make_program(config.time_program, instance, rate)
    if marker_instance is not None:
        marker_time = program.state(marker_instance)
        marker_hold = program.value(
            "marker_hold",
            marker_time.n,
            at=marker_time.next.point,
        )
        program.commit(marker_time.next, marker_hold)
    step_dt = config.step_dt
    if step_dt is not None:
        program.step_strategy(FixedDt(dt=float(step_dt)))
    else:
        program.step_strategy(AdaptiveCFL(cfl=float(config.cfl)))
    case.program(program)
    attach_case_diagnostics(case, tracer, program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=_sine_initial(frame, config.wave),
            projection=ConservativeCellAverage(),
        )
    )
    if marker_instance is not None:
        case.initials.add(
            InitialCondition(
                state=marker_instance,
                value=_half_marker(frame),
                projection=ConservativeCellAverage(),
            )
        )
    return _Authoring(
        case=case,
        instance=instance,
        marker_instance=marker_instance,
        block=tracer,
        frame=frame,
        config=config,
        program=program,
    )


def _cells(config: Tr01Config) -> tuple[int, ...]:
    return (config.mesh_cells,) * config.dim


def layout_for(authored: _Authoring):
    """Return the public Uniform or AMR layout, or raise a precise blocker."""
    config = authored.config
    if config.layout in UNIFORM_LAYOUTS:
        return uniform_periodic_layout(authored.frame, _cells(config))
    if config.layout == "A-DP":
        raise AuthoringPending(
            "A-DP requires an independently advected marker with scheduled regrid; "
            "TR-01 public SSPRK2/RK4 advances one tracer state and AM-01 hold is static"
        )

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

    transfer = AMRTransfer()
    transfer.state(authored.instance, StateTransfer())
    buffer_cells = 3 if config.reconstruction == "weno5z" else 2
    if config.layout in {"A-S0", "A-S2"}:
        if authored.marker_instance is None:
            raise AuthoringPending(
                f"{config.layout}: static marker block was not authored"
            )
        transfer.state(authored.marker_instance, StateTransfer())
        threshold = authored.case.param(
            RuntimeParam("tr01_refine_marker", default=0.5)
        )
        rules = (
            Tag(ValueExpr(authored.marker_instance) > authored.case.value(threshold)),
            Buffer(cells=buffer_cells),
        )
        regrid = AMRRegrid.frozen()
    elif config.layout == "A-DT":
        refine = authored.case.param(
            RuntimeParam("tr01_refine_q", default=float(_exact.Q0) + 0.3 * _exact.EPS)
        )
        coarsen = authored.case.param(
            RuntimeParam("tr01_coarsen_q", default=float(_exact.Q0) + 0.15 * _exact.EPS)
        )
        rules = (
            Tag(ValueExpr(authored.instance) > authored.case.value(refine)),
            Coarsen(ValueExpr(authored.instance) < authored.case.value(coarsen)),
            Buffer(cells=buffer_cells),
        )
        regrid = AMRRegrid(schedule=every(2, clock=authored.program.clock))
    else:
        raise AuthoringPending(f"{config.layout}: unknown AMR layout")
    execution = (
        AMRExecution.subcycled((AMRClockRelation(0, 1, 2),))
        if config.layout == "A-S2"
        else AMRExecution.synchronous()
    )
    patch = PatchLayout()
    if config.block_size is not None:
        patch = PatchLayout(
            distribute_coarse=True,
            coarse_max_grid=int(config.block_size),
        )
    return AMR(
        grid=CartesianGrid(
            frame=authored.frame,
            cells=_cells(config),
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


def _canonical_3d(n_cells: int) -> Tr01Config:
    return Tr01Config(
        dim=3,
        velocity=tuple(float(v) for v in _exact.A),
        wave=tuple(float(v) for v in _exact.K),
        n_cells=int(n_cells),
        family="global",
        reconstruction="weno5z",
        label="canonical",
    )


def resolve_config_id(config_id: str, n_cells: int = 16) -> Tr01Config:
    """Return a named configuration. ``canonical`` is the 3-d §9.2 cube only."""
    count = int(n_cells)
    if config_id == "canonical":
        return _canonical_3d(count)
    if config_id == "restriction_1d":
        return Tr01Config(
            dim=1,
            velocity=(1.0,),
            wave=(1.0,),
            n_cells=count,
            family="global",
            reconstruction="weno5z",
            label="restriction_1d",
        )
    if config_id == "restriction_2d":
        return Tr01Config(
            dim=2,
            velocity=(1.0, 1.0),
            wave=(1.0, 2.0),
            n_cells=count,
            family="global",
            reconstruction="weno5z",
            label="restriction_2d",
        )
    if config_id == "tvd_vanleer":
        return Tr01Config(
            dim=1,
            velocity=(1.0,),
            wave=(1.0,),
            n_cells=count,
            family="tvd",
            reconstruction="vanleer",
            label="tvd_vanleer",
        )
    raise NativeUnavailable(f"unknown TR-01 config_id {config_id!r}")


def resolve_config(request, config_id: str | None = None) -> Tr01Config:
    """Dispatch a CampaignRequest onto one exact-rank configuration."""
    dim = int(request.pops_native_dim)
    defaults = {1: "restriction_1d", 2: "restriction_2d", 3: "canonical"}
    chosen = config_id or defaults.get(dim)
    if chosen is None:
        raise NativeUnavailable(f"unsupported request dim {dim}")
    config = resolve_config_id(chosen, n_cells=int(request.min_resolution or 16))
    if config.dim != dim:
        raise NativeUnavailable(
            f"config {chosen!r} dim {config.dim} does not match request dim {dim}; "
            "canonical 3-d is not substituted"
        )
    return config


def _launched_native_dim() -> int | None:
    raw = os.environ.get("POPS_NATIVE_DIM")
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def require_exact_rank(request, launched_dim: int | None = None) -> int:
    """Refuse a silent dimension substitution."""
    launched = _launched_native_dim() if launched_dim is None else launched_dim
    required = int(request.pops_native_dim)
    if launched != required:
        raise NativeUnavailable(
            f"POPS_NATIVE_DIM={launched!r} does not match request dim {required}; "
            "no fallback to another native extension"
        )
    return required


def _require_native_dim3() -> None:
    launched = _launched_native_dim()
    if launched != REQUIRED_NATIVE_DIM:
        raise NativeUnavailable(
            f"TR-01 requires POPS_NATIVE_DIM={REQUIRED_NATIVE_DIM} "
            f"(got {launched!r}); no 1-d/2-d fallback"
        )


def _confirm_selected_dim(dim: int) -> None:
    from pops._native_selector import selected_native_dimension

    selected = selected_native_dimension()
    if selected != int(dim):
        raise NativeUnavailable(
            f"selected native dimension is {selected!r}, not {dim}; "
            "the loaded artifact is not the requested exact-rank leaf"
        )


def _native_unavailable_reason() -> str | None:
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def build_case(n_cells: int = 16) -> pops.Case:
    """Author the canonical 3-d cube. Does not compile or run."""
    return author(_canonical_3d(n_cells)).case


def resolve_plan(n_cells: int = 16):
    """Validate and resolve the canonical 3-d cube."""
    return resolve_plan_for(_canonical_3d(n_cells))


def resolve_plan_for(config: Tr01Config):
    """Validate and resolve one configuration. Does not compile or call pops.run."""
    try:
        authored = author(config)
        return resolve_case(authored.case, layout=layout_for(authored))
    except AuthoringPending:
        raise
    except Exception as exc:
        raise AuthoringPending(
            f"{config.config_id} resolve failed: {type(exc).__name__}: {exc}"
        ) from exc


def _add(rows: list[Tr01Config], **kwargs) -> None:
    rows.append(Tr01Config(**kwargs))


def variant_catalog(*, dim: int | None = None) -> tuple[Tr01Config, ...]:
    """Obligatory v1.5 variants. 3-d n≥256 is excluded as memory-unsafe here."""
    rows: list[Tr01Config] = []
    if dim in (None, 1):
        for speed in (1.0, -1.0):
            for n_cells in (16, 32, 64, 128):
                _add(
                    rows,
                    dim=1,
                    velocity=(speed,),
                    wave=(1.0,),
                    n_cells=n_cells,
                    family="global",
                    label="restriction_1d" if speed == 1.0 else "variant",
                )
        for periods in (2, 4):
            _add(
                rows,
                dim=1,
                velocity=(1.0,),
                wave=(1.0,),
                n_cells=64,
                periods=periods,
                family="periods",
            )
        _add(
            rows,
            dim=1,
            velocity=(1.0,),
            wave=(1.0,),
            n_cells=32,
            layout="U-F",
            family="uniform_fine",
        )
        for layout in AMR_LAYOUTS:
            _add(
                rows,
                dim=1,
                velocity=(1.0,),
                wave=(1.0,),
                n_cells=32,
                layout=layout,
                family="amr",
            )
        for block in (8, 16, 32, 64):
            _add(
                rows,
                dim=1,
                velocity=(1.0,),
                wave=(1.0,),
                n_cells=64,
                layout="A-S0",
                block_size=block,
                family="blocks",
            )
    if dim in (None, 2):
        for velocity in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.37)):
            for n_cells in (16, 32, 64, 128):
                label = "restriction_2d" if velocity == (1.0, 1.0) else "variant"
                _add(
                    rows,
                    dim=2,
                    velocity=velocity,
                    wave=(1.0, 2.0),
                    n_cells=n_cells,
                    family="global",
                    label=label,
                )
        for periods in (2, 4):
            _add(
                rows,
                dim=2,
                velocity=(1.0, 1.0),
                wave=(1.0, 2.0),
                n_cells=32,
                periods=periods,
                family="periods",
            )
        _add(
            rows,
            dim=2,
            velocity=(1.0, 1.0),
            wave=(1.0, 2.0),
            n_cells=32,
            layout="U-F",
            family="uniform_fine",
        )
        for layout in AMR_LAYOUTS:
            _add(
                rows,
                dim=2,
                velocity=(1.0, 1.0),
                wave=(1.0, 2.0),
                n_cells=16,
                layout=layout,
                family="amr",
            )
        for block in (8, 16, 32):
            _add(
                rows,
                dim=2,
                velocity=(1.0, 1.0),
                wave=(1.0, 2.0),
                n_cells=32,
                layout="A-S0",
                block_size=block,
                family="blocks",
            )
    if dim in (None, 3):
        wave3 = (1.0, 2.0, 3.0)
        for velocity in (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.37, 0.61),
        ):
            for n_cells in (16, 32, 64):
                _add(
                    rows,
                    dim=3,
                    velocity=velocity,
                    wave=wave3,
                    n_cells=n_cells,
                    family="global",
                )
        for n_cells in (16, 32, 64, 128):
            _add(
                rows,
                dim=3,
                velocity=(1.0, 1.0, 1.0),
                wave=wave3,
                n_cells=n_cells,
                family="global",
                label="canonical",
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
        for layout in AMR_LAYOUTS:
            _add(
                rows,
                dim=3,
                velocity=(1.0, 1.0, 1.0),
                wave=wave3,
                n_cells=16,
                layout=layout,
                family="amr",
            )
        for block in (8, 16, 32):
            _add(
                rows,
                dim=3,
                velocity=(1.0, 1.0, 1.0),
                wave=wave3,
                n_cells=16,
                layout="A-S0",
                block_size=block,
                family="blocks",
            )
    return tuple(row for row in rows if dim is None or row.dim == dim)


def variant_status(config: Tr01Config) -> dict[str, Any]:
    """Public AMR/uniform status. Never claim an unauthored layout is executable."""
    if config.layout == "A-DP":
        return {
            "status": "required_failure",
            "blocker": (
                "A-DP requires an independently advected marker with scheduled regrid; "
                "TR-01 public SSPRK2/RK4 advances one tracer state and AM-01 hold is static"
            ),
        }
    try:
        resolve_plan_for(config)
    except AuthoringPending as exc:
        return {"status": "required_failure", "blocker": str(exc)}
    except Exception as exc:
        return {
            "status": "required_failure",
            "blocker": f"{config.layout} resolve failed: {type(exc).__name__}: {exc}",
        }
    return {"status": "authoring_ok", "blocker": None}


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


def require_live_execution_space(request) -> str:
    """Refuse a Serial claim on an OpenMP default space. Unknown is not Serial proof."""
    facts = _live_runtime_facts()
    backend = str(facts.get("kokkos_backend") or "unknown")
    requested = str(request.execution_space)
    if requested == "KokkosSerial" and backend in {
        "OpenMP",
        "KokkosOpenMP",
        "Cuda",
        "KokkosCuda",
    }:
        raise NativeUnavailable(
            f"KokkosSerial requested but live backend is {backend}; Serial not-run"
        )
    if requested == "KokkosOpenMP" and backend not in {
        "OpenMP",
        "KokkosOpenMP",
        "unknown",
    }:
        raise NativeUnavailable(
            f"KokkosOpenMP requested but live backend is {backend}"
        )
    return requested


def campaign_run_fields(request, config: Tr01Config, n_cells: int, t_end: float) -> dict[str, Any]:
    """Live ranks/threads/space when the native world is present. No manifest defaults."""
    from verification.pops_verify.mpi_world import native_world_size

    count = int(n_cells)
    dim = int(config.dim)
    mpi_on = request.mpi_mode == "on"
    facts = _live_runtime_facts()
    live_ranks = native_world_size(required=False)
    if live_ranks is not None:
        mpi_ranks = int(live_ranks)
    elif mpi_on:
        mpi_ranks = int(request.resources.mpi_ranks)
    else:
        mpi_ranks = 1
    concurrency = int(facts.get("kokkos_concurrency") or 0)
    if concurrency > 0:
        threads = concurrency
    else:
        threads = int(request.resources.omp_threads)
    recorded_space = _space_from_backend(
        str(facts.get("kokkos_backend") or "unknown"),
        request.execution_space,
    )
    step_dt = config.step_dt
    cfl = config.cfl
    if cfl is None and step_dt is not None:
        cfl = float(step_dt) * float(config.mesh_cells)
    return {
        "compiler": default_cxx() or os.environ.get("CXX", "c++"),
        "build_type": os.environ.get("CMAKE_BUILD_TYPE", "Release"),
        "precision": "float64",
        "kokkos_execution_space": recorded_space,
        "mpi_enabled": mpi_on,
        "mpi_library": (
            (os.environ.get("POPS_MPI_LIBRARY") or "mpich") if mpi_on else "none"
        ),
        "mpi_thread_level_requested": "MPI_THREAD_SINGLE" if mpi_on else "none",
        "mpi_thread_level_provided": "MPI_THREAD_SINGLE" if mpi_on else "none",
        "hdf5_collective_enabled": False,
        "mpi_ranks": mpi_ranks,
        "omp_threads_per_rank": threads,
        "gpus": 0,
        "resolution": [count] * dim,
        "block_size": [int(config.block_size or count)] * dim,
        "amr_total_levels": 2 if config.layout in AMR_LAYOUTS else 1,
        "refinement_ratio": 2,
        "subcycling": config.layout == "A-S2",
        "time_program": config.time_program,
        "cfl": float(cfl if cfl is not None else 0.0),
        "final_time": float(t_end),
    }


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


def _oracle_cell_averages(config: Tr01Config, t: float) -> np.ndarray:
    lo, hi = _exact.cell_bounds_nd(config.mesh_cells, config.dim)

    def _u(*args):
        *coords, time = args
        return _exact.exact_sine_nd(coords, time, a=config.velocity, k=config.wave)

    return analytic_cell_averages(_u, lo, hi, t)


def fourier_mode_coefficient(field, coords, volumes, wave) -> complex:
    """Complex Fourier coefficient of ``field`` on mode ``wave`` using cell volumes."""
    arrays = [np.asarray(axis, dtype=np.float64) for axis in coords]
    weights = np.asarray(volumes, dtype=np.float64)
    values = np.asarray(field, dtype=np.float64)
    if len(arrays) != len(wave):
        raise ValueError("coords and wave must have the same rank")
    phase = np.zeros(values.shape, dtype=np.float64)
    for axis, mode in zip(arrays, wave, strict=True):
        phase = phase + float(mode) * np.asarray(axis, dtype=np.float64)
    kernel = np.exp(-2.0j * np.pi * phase)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("volumes must be positive")
    return complex(np.sum(values * kernel * weights) / total)


def field_diagnostics(field, oracle, volumes, *, config: Tr01Config) -> dict[str, Any]:
    """Norms, Fourier phase/amplitude on k, and mass from native vs oracle."""
    errors = reference_errors(field, oracle, volumes)
    coords, _ = _exact.uniform_cell_mesh_nd(config.mesh_cells, config.dim)
    coeff_num = fourier_mode_coefficient(field, coords, volumes, config.wave)
    coeff_ref = fourier_mode_coefficient(oracle, coords, volumes, config.wave)
    if abs(coeff_ref) == 0.0:
        phase = None
    else:
        phase = float(np.angle(coeff_num / coeff_ref))
    amp_num = 2.0 * abs(coeff_num)
    amplitude_loss = 1.0 - amp_num / float(_exact.EPS)
    integral = float(np.sum(np.asarray(field) * np.asarray(volumes)))
    volume = float(np.sum(volumes))
    mass_error = integral - float(_exact.Q0) * volume
    return {
        "l1": float(errors.l1),
        "l2": float(errors.l2),
        "linf": float(errors.linf),
        "phase_error": phase,
        "amplitude_loss": float(amplitude_loss),
        "integral": integral,
        "mass_error": float(mass_error),
    }


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
    return {
        "component_catalog_digest": str(catalog),
        "native_header_signature": str(header),
        "native_variant_manifest_digest": _sha256_file(manifests[0]),
    }


def _write_run_provenance(output_dir: Path | None, request, config: Tr01Config, t_end: float):
    if request is None:
        return None
    document = collect_provenance(
        "TR-01",
        pops_native_dim=config.dim,
        dimension=config.dim,
        nodes=int(getattr(request.resources, "nodes", 1) or 1),
        run=campaign_run_fields(request, config, config.n_cells, t_end),
        doctor_ok=True,
        **_live_provenance_digests(),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_provenance(output_dir / f"provenance_n{int(config.n_cells)}.json", document)
        write_provenance(output_dir / "provenance.json", document)
    return document


def _legacy_run_fields(n_cells: int, t_end: float) -> dict[str, Any]:
    class _Legacy:
        execution_space = "KokkosSerial"
        mpi_mode = "off"

        class resources:
            mpi_ranks = 1
            omp_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
            nodes = 1

    return campaign_run_fields(
        _Legacy(),
        _canonical_3d(n_cells),
        n_cells,
        t_end,
    )


def _execute(config: Tr01Config, *, request=None, output_dir: Path | None = None) -> dict[str, Any]:
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    if request is not None:
        require_exact_rank(request)
        require_live_execution_space(request)
        if request.pops_native_dim != config.dim:
            raise NativeUnavailable("request dim does not match resolved config")
    elif config.dim != REQUIRED_NATIVE_DIM:
        raise NativeUnavailable(
            f"TR-01 requires POPS_NATIVE_DIM={REQUIRED_NATIVE_DIM} "
            f"(got {config.dim!r}); no 1-d/2-d fallback"
        )
    else:
        _require_native_dim3()
    if config.layout in AMR_LAYOUTS:
        status = variant_status(config)
        if status["status"] != "authoring_ok":
            raise AuthoringPending(status["blocker"])
    authored = author(config)
    plan = resolve_case(authored.case, layout=layout_for(authored))
    if getattr(plan, "resolved_dimension", None) != config.dim:
        raise NativeUnavailable(
            f"resolved dimension is {getattr(plan, 'resolved_dimension', None)!r}, "
            f"not {config.dim}"
        )
    artifact = pops.compile(plan)
    _confirm_selected_dim(config.dim)
    mpi_mode = "off" if request is None else request.mpi_mode
    simulation = bind_public(artifact, mpi_mode=mpi_mode)
    pops.run(simulation, t_end=float(config.t_end), max_steps=MAX_STEPS)
    field = _unpack_field(
        simulation.state_global("tracer"), config.mesh_cells, config.dim
    )
    oracle = _oracle_cell_averages(config, config.t_end)
    _, volumes = _exact.uniform_cell_mesh_nd(config.mesh_cells, config.dim)
    diagnostics = field_diagnostics(field, oracle, volumes, config=config)
    n_steps = int(getattr(simulation, "n_accepted_steps", 0) or 0)
    diagnostics["conservation_tolerance"] = float(
        conservation_tolerance(
            1.0, abs_tol=1.0e-12, rel_tol=1.0e-12, n_updates=max(n_steps, 1), c=100.0
        )
    )
    diagnostics["conservation_ok"] = (
        abs(diagnostics["mass_error"]) <= diagnostics["conservation_tolerance"]
    )
    if request is not None:
        fields = campaign_run_fields(request, config, config.n_cells, config.t_end)
        _write_run_provenance(output_dir, request, config, config.t_end)
    else:
        fields = _legacy_run_fields(config.n_cells, config.t_end)
        if output_dir is not None:
            document = collect_provenance(
                "TR-01",
                pops_native_dim=REQUIRED_NATIVE_DIM,
                dimension=REQUIRED_NATIVE_DIM,
                nodes=1,
                run=fields,
                doctor_ok=True,
                **_live_provenance_digests(),
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            write_provenance(output_dir / f"provenance_n{int(config.n_cells)}.json", document)
            write_provenance(output_dir / "provenance.json", document)
    payload = dict(fields)
    payload.update(
        {
            "field": field,
            "result": field,
            "program_bytes": program_bytes_from_artifact(artifact),
            "oracle": oracle,
            "coordinates": _exact.uniform_cell_mesh_nd(config.mesh_cells, config.dim)[0],
            "volumes": volumes,
            "diagnostics": diagnostics,
            "config_id": config.config_id,
            "label": config.label,
            "dimension": config.dim,
            "source": "native",
        }
    )
    return payload


def run_native(
    n_cells: int | None = None,
    t_end: float | None = None,
    *,
    request=None,
    output_dir: Path | None = None,
    config_id: str | None = None,
):
    """Compile, bind, and run one exact-rank configuration.

    Without ``request`` this remains the canonical 3-d cube and returns the
    field array. With ``request`` the return is RUN_FIELDS plus native science.
    """
    if request is not None:
        config = resolve_config(request, config_id=config_id)
        if n_cells is not None:
            config = config.replace(n_cells=int(n_cells))
        if t_end is not None and abs(float(t_end) - config.t_end) > 1.0e-15:
            raise NativeUnavailable("t_end override would silently replace the config period")
        return _execute(config, request=request, output_dir=output_dir or request.output_dir)
    count = 16 if n_cells is None else int(n_cells)
    if t_end is None or abs(float(t_end) - T_END) <= 1.0e-15:
        payload = _execute(_canonical_3d(count), output_dir=output_dir)
        return payload["field"]
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    _require_native_dim3()
    authored = author(_canonical_3d(count))
    plan = resolve_case(
        authored.case,
        layout=uniform_periodic_layout(authored.frame, (count, count, count)),
    )
    artifact = pops.compile(plan)
    _confirm_selected_dim(REQUIRED_NATIVE_DIM)
    simulation = bind_public(artifact)
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = _unpack_field(simulation.state_global("tracer"), count, 3)
    if output_dir is not None:
        document = collect_provenance(
            "TR-01",
            pops_native_dim=REQUIRED_NATIVE_DIM,
            dimension=REQUIRED_NATIVE_DIM,
            nodes=1,
            run=_legacy_run_fields(count, float(t_end)),
            doctor_ok=True,
            **_live_provenance_digests(),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_provenance(output_dir / f"provenance_n{count}.json", document)
        write_provenance(output_dir / "provenance.json", document)
    return field


def run_order_campaign(
    resolutions=RESOLUTIONS,
    *,
    t_end: float | None = None,
    output_dir: Path | None = None,
    request=None,
    config_id: str | None = None,
) -> dict:
    """Native Δx series. Fewer than four resolutions cannot claim order."""
    steps = tuple(int(n) for n in resolutions)
    if len(steps) < 4:
        raise ValueError("TR-01 requires at least four resolutions")
    if any(n <= 0 for n in steps):
        raise ValueError("resolutions must be positive")
    if request is not None:
        base = resolve_config(request, config_id=config_id)
    else:
        _require_native_dim3()
        base = _canonical_3d(steps[0])
    if t_end is not None and abs(float(t_end) - base.t_end) > 1.0e-15:
        raise NativeUnavailable("order campaign t_end must match the configuration period")
    fields = {}
    oracles = {}
    volumes = {}
    diagnostics = {}
    l1 = []
    l2 = []
    linf = []
    for n_cells in steps:
        config = base.replace(n_cells=int(n_cells))
        payload = _execute(config, request=request, output_dir=output_dir)
        fields[n_cells] = payload["field"]
        oracles[n_cells] = payload["oracle"]
        volumes[n_cells] = payload["volumes"]
        diagnostics[n_cells] = payload["diagnostics"]
        l1.append(float(payload["diagnostics"]["l1"]))
        l2.append(float(payload["diagnostics"]["l2"]))
        linf.append(float(payload["diagnostics"]["linf"]))
    return {
        "source": "native",
        "case_id": "TR-01",
        "pops_native_dim": base.dim,
        "dimension": base.dim,
        "label": base.label,
        "family": base.family,
        "reconstruction": base.reconstruction,
        "time_program": base.time_program,
        "dt_scaling": base.dt_scaling,
        "velocity": tuple(base.velocity),
        "wave": tuple(base.wave),
        "t_end": float(base.t_end),
        "resolutions": steps,
        "spacings": tuple(1.0 / float(n) for n in steps),
        "l1": tuple(l1),
        "l2": tuple(l2),
        "linf": tuple(linf),
        "threshold": ORDER_THRESHOLD,
        "fields": fields,
        "oracles": oracles,
        "volumes": volumes,
        "diagnostics": diagnostics,
    }


def default_temporal_dts(n_cells: int = 64, *, cfl_max: float = 0.45) -> tuple[float, ...]:
    """Future temporal defaults start at CFL ≤ 0.45 (dt ≤ 0.005 at N=64)."""
    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    dt0 = min(0.005, float(cfl_max) / float(count))
    return tuple(dt0 / (2**index) for index in range(4))


def run_temporal_campaign(
    dts,
    *,
    n_cells: int = 64,
    output_dir: Path | None = None,
    request=None,
    config_id: str | None = None,
) -> dict:
    """§9.4 fixed-grid temporal series: dt, dt/2, dt/4, dt/8 (dt/16 optional)."""
    steps = tuple(float(value) for value in dts)
    if len(steps) < 4:
        raise ValueError("TR-01 temporal campaign requires at least four dt values")
    if any(value <= 0.0 for value in steps):
        raise ValueError("dts must be positive")
    count = int(n_cells)
    if request is not None:
        base = resolve_config(request, config_id=config_id)
    else:
        _require_native_dim3()
        base = _canonical_3d(count)
    fields = {}
    oracles = {}
    volumes = {}
    diagnostics = {}
    l1 = []
    l2 = []
    linf = []
    factors = []
    for index, dt in enumerate(steps):
        factor = 2**index
        factors.append(factor)
        config = base.replace(
            n_cells=count,
            dt=float(dt),
            family="temporal",
            dt_scaling="fixed",
            reconstruction=base.reconstruction,
        )
        payload = _execute(config, request=request, output_dir=output_dir)
        fields[factor] = payload["field"]
        oracles[factor] = payload["oracle"]
        volumes[factor] = payload["volumes"]
        diagnostics[factor] = payload["diagnostics"]
        l1.append(float(payload["diagnostics"]["l1"]))
        l2.append(float(payload["diagnostics"]["l2"]))
        linf.append(float(payload["diagnostics"]["linf"]))
    return {
        "source": "native",
        "case_id": "TR-01",
        "pops_native_dim": base.dim,
        "dimension": base.dim,
        "label": base.label,
        "family": "temporal",
        "reconstruction": base.reconstruction,
        "time_program": base.time_program,
        "dt_scaling": "fixed",
        "velocity": tuple(base.velocity),
        "wave": tuple(base.wave),
        "t_end": float(base.t_end),
        "n_cells": count,
        "dts": steps,
        "resolutions": tuple(factors),
        "spacings": steps,
        "l1": tuple(l1),
        "l2": tuple(l2),
        "linf": tuple(linf),
        "threshold": ORDER_THRESHOLD,
        "fields": fields,
        "oracles": oracles,
        "volumes": volumes,
        "diagnostics": diagnostics,
    }
