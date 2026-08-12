"""Native capability report and route-row builders (ADC-619 split).

The native side of the capability layer: :class:`NativeCapabilityReport` (the
Python value object for ``_pops.capability_report()``), the lazy ``_pops`` bridges
(``_module_capabilities`` / ``_native_capability_report_from_extension``), the flat
route-row builders (``_support_rows`` / ``_inventory_rows`` / ``_row``) and the
public ``native_capability_report`` / ``native_capability_matrix`` entry points.
Split out of ``_capabilities`` for the 500-line cap; ``pops._capabilities``
re-exports every name here. ``_pops`` is imported LAZILY so this module stays
importable without a compiled extension.
"""

from __future__ import annotations

import json
import importlib
from collections.abc import Mapping
from typing import Any

from pops._capabilities_common import (
    CapabilityRouteMatrix,
    CapabilityRouteRow,
    _flag_value,
    _status_from_flag,
    _unsupported_error,
)


class NativeCapabilityReportError(RuntimeError):
    """A loaded native extension cannot supply a valid capability report."""


def _native_extension() -> Any:
    """Load the native extension, returning ``None`` only when it is truly absent.

    An import error *inside* an installed extension is evidence of a broken native route, not an
    optional-source-only installation.  Preserve it so callers never turn a failed native report
    into an ``unknown`` capability matrix.
    """
    for name in ("_pops", "pops._pops"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    return None


def _module_capabilities(target: str = "module") -> Any:
    """The C++ authoritative capability dict, or ``None`` for a source-only install.

    Lazily imports ``_pops`` (top-level then ``pops._pops``, mirroring the codegen toolchain) so the
    module import graph stays acyclic and the catalog walk works with no compiled module present.
    ``None`` means the extension is absent.  A loaded extension must expose and successfully call
    ``module_capabilities``; an old, broken, or malformed native route is an error, never an
    indistinguishable ``unknown`` fallback.
    """
    mod = _native_extension()
    if mod is None:
        return None
    fn = getattr(mod, "module_capabilities", None)
    if not callable(fn):
        raise NativeCapabilityReportError(
            "loaded _pops extension does not expose callable module_capabilities()"
        )
    try:
        payload = fn(target)
        if not isinstance(payload, Mapping):
            raise TypeError("module_capabilities() must return a mapping")
        return dict(payload)
    except Exception as exc:
        raise NativeCapabilityReportError(
            "_pops.module_capabilities(%r) failed or returned a malformed mapping" % target
        ) from exc


def _native_capability_report_from_extension(target: str = "module") -> Any:
    """Return ``_pops.capability_report(target)`` as :class:`NativeCapabilityReport`, or ``None``."""
    mod = _native_extension()
    if mod is None:
        return None
    fn = getattr(mod, "capability_report", None)
    if not callable(fn):
        raise NativeCapabilityReportError(
            "loaded _pops extension does not expose callable capability_report()"
        )
    try:
        payload = fn(target)
        if not isinstance(payload, Mapping):
            raise TypeError("capability_report() must return a mapping")
        return NativeCapabilityReport.from_dict(dict(payload))
    except Exception as exc:
        raise NativeCapabilityReportError(
            "_pops.capability_report(%r) failed or returned a malformed mapping" % target
        ) from exc


def native_capability_report(
    target: str = "module", *, flags: Any = None, source: Any = None
) -> Any:
    """Return the structured native capability report (ADC-591).

    With a current ``_pops`` build, the native envelope comes from C++
    ``capability_report(target)`` and is augmented only by narrower public Python contract rows that
    are not expressible as module-wide C++ flags. ``flags`` is the manifest fallback path for
    already-compiled artifacts: it builds the same stable envelope from the manifest support flags
    and the Python inventory rows, without loading or recompiling the artifact.
    """
    if flags is None:
        report = _native_capability_report_from_extension(target)
        if report is not None:
            existing = {row.feature for row in report.routes}
            report.routes.extend(
                row
                for row in _python_contract_rows(report.capabilities, "python-contract")
                if row.feature not in existing
            )
            return report
        flags = _module_capabilities(target)
        source = source or ("native" if flags is not None else "source-only")
    else:
        source = source or "manifest"
    rows = _support_rows(flags, source) + _inventory_rows(flags, source)
    caps = dict(flags or {})
    return NativeCapabilityReport(
        schema_version=0,
        abi_version=int(caps.get("abi_version", 0) or 0),
        target=target,
        abi_key=None,
        platform="unknown",
        capabilities=caps,
        runtime={},
        routes=rows,
    )


def _feature_layout(feature: str) -> str:
    if feature == "supports_uniform":
        return "uniform"
    if feature == "supports_amr":
        return "amr"
    return "uniform|amr"


def _feature_backend(feature: str) -> str:
    if feature in ("supports_stride", "supports_amr"):
        return "production"
    return "module"


def _feature_platform(feature: str) -> str:
    if feature in ("supports_mpi", "supports_custom_communicator"):
        return "mpi"
    if feature == "supports_gpu":
        return "gpu"
    return "host"


def _flag_error_message(feature: str) -> str:
    requests = {
        "supports_amr": (
            "layout=AMR",
            "layout=Uniform or backend='production' target='amr_system'",
            "use layout=Uniform or compile with backend='production' target='amr_system'",
        ),
        "supports_mpi": (
            "platform=MPI",
            "serial/OpenMP build",
            "rebuild _pops with POPS_USE_MPI=ON",
        ),
        "supports_gpu": (
            "platform=GPU",
            "host CPU platform",
            "use the delivered host/serial route or a separately proved CUDA/HIP artifact",
        ),
        "supports_stride": (
            "strided cell access",
            "backend='production'",
            "compile with backend='production'",
        ),
        "supports_partial_imex_mask": (
            "partial IMEX mask",
            "typed local implicit Program primitive where implemented",
            "author an explicit typed Program; AMR currently has no implicit-source primitive",
        ),
        "supports_custom_communicator": (
            "communicator != MPI_COMM_WORLD",
            "MPI_COMM_WORLD or serial",
            "use ExecutionContext.mpi_world() or a serial context",
        ),
    }
    requested, available, alternative = requests.get(
        feature, (feature, "no route in this build", None)
    )
    return _unsupported_error(requested=requested, available=available, alternative=alternative)


class NativeCapabilityReport:
    """Versioned structured native capability report (ADC-591).

    This is the Python value object for ``_pops.capability_report()``. Pretty route matrices and
    legacy ``module_capabilities()`` dicts are projections of this object. ``routes`` is a list of
    :class:`CapabilityRouteRow` instances, each carrying a status and reason directly, so tests and
    validators do not parse formatted strings.
    """

    def __init__(
        self,
        *,
        schema_version: Any,
        abi_version: Any,
        target: Any,
        abi_key: Any,
        platform: Any,
        capabilities: Any,
        runtime: Any,
        routes: Any,
    ) -> None:
        self.schema_version = int(schema_version)
        self.abi_version = int(abi_version)
        self.target = target
        self.abi_key = abi_key
        self.platform = platform
        self.capabilities = dict(capabilities or {})
        self.runtime = dict(runtime or {})
        self.routes = list(routes)

    @classmethod
    def from_dict(cls, payload: Any) -> NativeCapabilityReport:
        routes = [_route_from_native_dict(row) for row in payload.get("routes", [])]
        return cls(
            schema_version=payload.get("schema_version", 0),
            abi_version=payload.get(
                "abi_version", payload.get("capabilities", {}).get("abi_version", 0)
            ),
            target=payload.get("target", "module"),
            abi_key=payload.get("abi_key"),
            platform=payload.get("platform"),
            capabilities=payload.get("capabilities", {}),
            runtime=payload.get("runtime", {}),
            routes=routes,
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "abi_version": self.abi_version,
            "target": self.target,
            "abi_key": self.abi_key,
            "platform": self.platform,
            "capabilities": dict(self.capabilities),
            "runtime": dict(self.runtime),
            "routes": [row.to_dict() for row in self.routes],
        }

    def to_json(self, path: Any = None, *, indent: int = 2) -> Any:
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def route(self, feature: str) -> Any:
        for row in self.routes:
            if row.feature == feature:
                return row
        raise KeyError(feature)

    def __repr__(self) -> str:
        return "NativeCapabilityReport(schema=%r, abi=%r, target=%r, routes=%d)" % (
            self.schema_version,
            self.abi_version,
            self.target,
            len(self.routes),
        )

    def __str__(self) -> str:
        lines = [
            "native capability report (schema=%s, abi=%s, target=%s)"
            % (self.schema_version, self.abi_version, self.target)
        ]
        lines.append("  platform : %s" % self.platform)
        lines.append("  abi_key  : %s" % ((self.abi_key or "")[:12] or "none"))
        lines.append(
            "  runtime  : dimension=%s amr_refinement_ratio=%s selection=%s rank=%s "
            "precision=%s communicator=%s"
            % (
                self.runtime.get("dimension"),
                self.runtime.get("amr_refinement_ratio"),
                self.runtime.get("amr_refinement_ratio_selection"),
                self.runtime.get("amr_refinement_ratio_rank"),
                self.runtime.get("precision"),
                self.runtime.get("communicator"),
            )
        )
        lines.append("  routes   : %d structured row(s)" % len(self.routes))
        for row in self.routes:
            if row.status != "available":
                lines.append("    %-34s %-11s %s" % (row.feature, row.status, row.limitation))
        return "\n".join(lines)


def _route_from_native_dict(raw: Any) -> Any:
    status = raw.get("status", "unknown")
    requested = raw.get("requested") or raw.get("feature")
    available_route = raw.get("available_route") or "no native route"
    alternative = raw.get("alternative") or None
    limitation = raw.get("reason") or raw.get("limitation") or ""
    error = raw.get("error_message") or ""
    if status == "unavailable" and not error:
        error = _unsupported_error(
            requested=requested, available=available_route, alternative=alternative
        )
    return CapabilityRouteRow(
        raw.get("feature") or raw.get("route_id"),
        layout=raw.get("layout", "any"),
        backend=raw.get("backend", "any"),
        platform=raw.get("platform", "host"),
        mpi=raw.get("mpi", False),
        gpu=raw.get("gpu", False),
        status=status,
        limitation=limitation,
        error_message=error,
        source=raw.get("source", "native"),
        axis=raw.get("axis"),
        available_route=raw.get("available_route", ""),
        alternative=raw.get("alternative", ""),
    )


def _row(
    feature: str,
    *,
    layout: str = "any",
    backend: str = "any",
    platform: str = "host",
    flags: Any = None,
    flag: Any = None,
    mpi: Any = False,
    gpu: Any = False,
    limitation: str = "",
    requested: Any = None,
    available_route: Any = None,
    alternative: Any = None,
    source: str = "native",
    status: Any = None,
) -> Any:
    if status is None:
        status = _status_from_flag(flags, flag) if flag else "available"
    err = ""
    if status == "unavailable":
        err = _unsupported_error(
            requested=requested or feature,
            available=available_route or "no native route",
            alternative=alternative,
        )
    return CapabilityRouteRow(
        feature,
        layout=layout,
        backend=backend,
        platform=platform,
        mpi=mpi,
        gpu=gpu,
        status=status,
        limitation=limitation,
        error_message=err,
        source=source,
        available_route=available_route or "",
        alternative=alternative or "",
    )


def _python_contract_rows(flags: Any, source: str) -> list[Any]:
    """Routes whose exact public constraint is narrower than the module-level native flags."""

    mpi = bool(_flag_value(flags, "supports_mpi"))
    gpu = bool(_flag_value(flags, "supports_gpu"))
    return [
        _row(
            "boundary:prepared_transport",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            status="partial",
            limitation=(
                "one prepared 2D model-aware plan serves Uniform/AMR native and compiled "
                "transport boundaries; executable built-ins are periodic, extrapolation, "
                "constant/RuntimeParam fixed state, conservative device-side analytic "
                "(x,y,t,params) fixed state, model primitive-to-conservative fixed-state conversion, "
                "typed-role slip wall, and typed no-flux faces that extrapolate ghosts then zero "
                "the evaluated numerical flux before divergence/reflux; dynamic AMR regrid keeps internal "
                "coarse-fine ghosts under the prepared transfer authority on MPI ranks, with "
                "double-physical corners explicitly not required by dimension-split FV stencils; "
                "numerical resolution rejects every descriptor outside this executable envelope"
            ),
            source=source,
        ),
        _row(
            "boundary:characteristic_no_inflow",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=False,
            gpu=False,
            status="partial",
            limitation=(
                "2D Cartesian conservative constant/RuntimeParam fixed-reference no-inflow uses "
                "the exact compiled "
                "flux-Jacobian provider emitted by m.roe_from_jacobian() (1..16 components); "
                "the Kokkos kernel projects only outward-normal incoming modes, treats the "
                "scale-relative sonic subspace as neutral, preflights the real spectrum "
                "collectively, and rolls back ghosts on refusal; primitive/analytic references, "
                "runtime/field-dependent eigenstructure, sonic-error policy, 3D, polar/embedded "
                "geometry, and qualified MPI/GPU execution remain unavailable"
            ),
            requested="characteristic no-inflow/outflow transport boundary",
            available_route=(
                "Inflow(state=U, value=U_ref, characteristic=model_characteristic_no_inflow(U))"
            ),
            alternative=(
                "use fixed-state inflow/extrapolated outflow outside the qualified envelope"
            ),
            source=source,
        ),
        _row(
            "boundary:representation_conversion",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            status="partial",
            limitation=(
                "2D fixed-state primitive inflow may use the exact compiled block-model "
                "to_conservative provider; conservative-to-primitive recovery and arbitrary "
                "representation converters remain unavailable, and conversion does not invent "
                "a boundary admissibility projection"
            ),
            source=source,
        ),
        _row(
            "boundary:analytic_xtp",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            status="partial",
            limitation=(
                "2D conservative fixed-state inflow accepts data-only analytic ScalarExpr "
                "programs over typed coordinates, one exact logical Clock, and bound parameters; "
                "primitive per-point conversion and discrete state/field/input reads remain "
                "unavailable, analytic ghost depth may not exceed the normal domain extent, and "
                "axis-permuted periodic coordinates require a prepared coordinate map"
            ),
            source=source,
        ),
        _row(
            "boundary:post_riemann_flux",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=False,
            status="partial",
            limitation=(
                "one typed BoundaryFlux component transforms the already evaluated outward-normal "
                "face flux between the Riemann solve and divergence/reflux through the same "
                "prepared Uniform/AMR boundary plan; execution is currently a 2D Cartesian "
                "host-batch route, the ordinary Uniform route materializes face fields when this "
                "stage is selected, and no device-native or embedded/cut-cell metric ABI or "
                "high-level TransportBoundarySet convenience exists yet"
            ),
            requested="post-Riemann transport-boundary flux provider",
            available_route="PostRiemannFlux plus one qualified BoundaryFlux component",
            source=source,
        ),
        _row(
            "riemann:typed_failure_outcome",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            status="partial",
            limitation=(
                "Rusanov, HLL, HLLC, and Roe return one device-copyable FluxEvaluation with "
                "typed status, stability bound, reason code, requested/used/last solver identity, "
                "and attempt metadata; single-solver routes remain explicit and face failures are "
                "reduced into the owning transaction, while fallback counters and restart "
                "publication metadata are not yet wired"
            ),
            source=source,
        ),
        _row(
            "riemann:prepared_recovery_policy",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=False,
            gpu=False,
            status="partial",
            limitation=(
                "the typed public riemann.Recovery descriptor lowers one catalog-authenticated "
                "Roe -> HLL -> Rusanov -> reject PreparedRiemannRecoveryPolicy into Uniform and "
                "AMR Cartesian face kernels and records requested, used, last-attempted, "
                "first-cause, and attempt-count provenance; only typed candidate rejection "
                "advances, while polar geometry is refused and block/team counters, MPI fallback "
                "reduction, GPU qualification, restart metadata, and a benchmark gate remain"
            ),
            requested=("prepared Riemann recovery chain with requested/used solver diagnostics"),
            available_route=(
                "pops.numerics.riemann.Recovery(primary=Roe(), fallbacks=(HLL(), Rusanov()))"
            ),
            alternative=(
                "select one supported Riemann route explicitly and consume rejection through "
                "the step retry/failure policy"
            ),
            source=source,
        ),
        _row(
            "recovery:prepared_variable",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            status="partial",
            limitation=(
                "one block-prepared closed-form method returns a device-copyable "
                "RecoveryOutcome/RecoveryReport retaining selected and last-attempted method "
                "kinds across type erasure; System conservative-to-primitive and transactional "
                "analytic initial-state materialization plus Cartesian, polar, masked, and "
                "embedded-boundary face "
                "reconstruction consume publication permission before copying or flux "
                "evaluation; primitive-to-conservative setup conversion publishes only a finite "
                "candidate accepted by that same prepared inverse authority; accepted AMR "
                "regrid prolongation and restriction candidates pass the block-prepared inverse "
                "authority collectively before replacing live hierarchy state; AMR bootstrap "
                "commits, rematerialized history slots, and physical boundary traces use that "
                "same publication gate and roll back exactly on refusal; generated Program "
                "terminal commits validate every Uniform or AMR live-state candidate before the "
                "first multi-block copy, including endpoints assembled from model-local and "
                "coupled sources, with no implicit repair or fallback; the host Uniform "
                "get_primitive_state materializer additionally owns one exact-state and "
                "generation-qualified warm-start slot per local cell, publishes only complete "
                "batches, and invalidates every slot after a refused batch"
            ),
            source=source,
        ),
        _row(
            "recovery:complete_consumer_cutover",
            layout="uniform|amr",
            backend="none",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            status="unavailable",
            limitation=(
                "manual in-place Program writes, persistent warm starts outside the host Uniform "
                "diagnostic materializer (spatial kernels and AMR), cache restart, and the "
                "backend/performance matrix do not yet share one prepared recovery authority"
            ),
            requested="complete prepared variable-recovery consumer cutover",
            available_route=(
                "prepared closed-form recovery for System conservative-to-primitive and "
                "transactional analytic initial-state materialization plus spatial face "
                "reconstruction, fallible primitive-to-conservative setup conversion, and "
                "transactional AMR regrid prolongation/restriction, bootstrap/history, and "
                "physical boundary-trace publication, plus generated Program terminal commit "
                "validation for model-local and coupled-source endpoints, and exact-state "
                "generation-qualified warm starts for host Uniform primitive materialization"
            ),
            alternative=(
                "use generated Program candidate commits and the delivered recovery consumers, or "
                "implement the missing in-place-write, AMR/spatial warm-start, and cache/restart "
                "contracts"
            ),
            source=source,
        ),
        _row(
            "amr:cell_local_temporal_transport",
            layout="amr",
            backend="production",
            platform="host",
            mpi=False,
            gpu=False,
            status="partial",
            limitation=(
                "Program.cell_local_time and its generated AmrProgramContext route cover one "
                "serial host rank, one 2D block, one level, one owned box, one common cell rung, "
                "transport-only forward Euler and frozen attempt auxiliary fields with built-in "
                "periodic/Foextrap boundaries; the provider reuses the exact compiled AMR "
                "residual/face-flux closure "
                "and commits real conservative state plus four time-integrated face records per "
                "cell as one accepted transaction at the synchronization barrier; its exact "
                "contract includes model-owned transport parameters and the limiter/Riemann route; "
                "same-topology restart restores numerical state and exact clocks but intentionally "
                "invalidates the last-interval diagnostic flux ledger until another accepted step; "
                "prepared physical-boundary plans, heterogeneous rungs, multi-box/multilevel and "
                "coarse/fine ledgers, sources, MPI, GPU, regrid/rank-change rematerialization, "
                "checkpoint persistence of the diagnostic ledger and performance proof remain "
                "unavailable"
            ),
            requested="prepared cell-local scientific stage and space-time flux transaction",
            available_route=(
                "Program.cell_local_time plus the generated AmrProgramContext and native "
                "PreparedSameLevelTransportEulerStageFluxProvider in their exact bounded "
                "host/serial same-rung envelope"
            ),
            alternative=(
                "use the synchronous AMR Program route outside that envelope, or implement the "
                "missing prepared local-time provider family"
            ),
            source=source,
        ),
        _row(
            "amr:external_field_solver_v2",
            layout="amr",
            backend="component",
            platform="host",
            mpi=mpi,
            gpu=False,
            status="available",
            limitation=(
                "host float64 with hierarchy-selected AMR ratios; MPI requires both components "
                "to declare "
                "MPI_COMM_WORLD and "
                "a distributed coarse level; executable MPI qualification currently covers "
                "exactly two ranks with distributed L0/L1 and regrid rematerialization; "
                "embedded/cut-cell topology, dynamic boundaries, reaction terms, nonlinear/JVP "
                "solves and GPU execution remain explicit refusals"
            ),
            requested="external FieldSolver@2 on an AMR hierarchy",
            available_route=(
                "authenticated FieldTopology@2 + FieldSolver@2 composite hierarchy batch with "
                "metadata.level, binary coarse/fine coverage, one collective solve, exact "
                "materialization/report consensus and transactional candidate publication"
            ),
            alternative="",
            source=source,
        ),
        _row(
            "amr:field_coupled_rhs_jacvec",
            layout="amr",
            backend="none",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            status="unavailable",
            limitation=(
                "field-coupled rhs_jacvec has no level-qualified tangent-field provider ABI "
                "for AMR level > 0"
            ),
            requested="field_coupled rhs_jacvec on AMR level > 0",
            available_route="field_coupled rhs_jacvec on AMR level 0",
            alternative=(
                "use the level-0 route or implement a level-qualified tangent-field provider ABI"
            ),
            source=source,
        ),
        _row(
            "amr:shared_interface_implicit_jacvec_pair",
            layout="amr",
            backend="production",
            platform="host",
            mpi=False,
            gpu=False,
            status="partial",
            limitation=(
                "one generated Program compiles, binds and runs GMRES with the paired "
                "level_rhs_jacvec_pair matvec on every level of an exactly two-level frozen 2D "
                "AMR hierarchy in host/serial execution; the two interface participants may use "
                "one independent packed-vector carrier block, but dynamic hierarchy mutation, "
                "additional interfaces, mixed apply operators, MPI and GPU remain unavailable"
            ),
            available_route=(
                "generated host/serial GMRES solve with an authenticated two-sided shared-interface "
                "JVP on a frozen two-level 2D AMR hierarchy"
            ),
            alternative=(
                "use the proved frozen two-level host/serial route, or add explicit execution "
                "proof for dynamic hierarchies, additional interfaces, MPI or GPU"
            ),
            source=source,
        ),
        _row(
            "amr:source_implicit_program",
            layout="amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=False,
            status="partial",
            limitation=(
                "a generated IMEX Program executes one prepared LocalNewton solve over every "
                "active cell on a synchronous, dynamically regridded two-level 2D hierarchy; "
                "SolveOutcome/FailRun rollback is exact across covered and uncovered coarse "
                "cells, fine cells, clocks, topology and MPI ranks, but GPU qualification, "
                "subcycled local solves, field/global implicit coupling and performance evidence "
                "remain outside the proved envelope"
            ),
            available_route=(
                "generated Program local implicit source solve with LocalNewton and a consumed "
                "SolveOutcome on synchronous two-level 2D AMR"
            ),
            alternative=(
                "use the proved synchronous local-source route, or add an explicit capability "
                "and execution proof for subcycled, GPU, field-coupled or global implicit solves"
            ),
            source=source,
        ),
    ]


def _support_rows(flags: Any, source: Any) -> list:
    return [
        _row(
            "supports_uniform",
            layout="uniform",
            backend="module",
            platform="host",
            flags=flags,
            flag="supports_uniform",
            limitation="single-level Uniform layout",
            requested="layout=Uniform",
            available_route="layout=Uniform",
            source=source,
        ),
        _row(
            "supports_amr",
            layout="amr",
            backend="production",
            platform="host",
            flags=flags,
            flag="supports_amr",
            limitation=(
                "hierarchy depth and transition ratios are selected by the "
                "authenticated AMR hierarchy"
            ),
            requested="layout=AMR",
            available_route="backend='production' target='amr_system'",
            alternative="use Uniform or an AMR hierarchy with explicit transition ratios",
            source=source,
        ),
        _row(
            "supports_mpi",
            layout="uniform|amr",
            backend="production",
            platform="mpi",
            flags=flags,
            flag="supports_mpi",
            mpi=bool(_flag_value(flags, "supports_mpi")),
            limitation="MPI is available only when _pops is built with POPS_USE_MPI=ON",
            requested="platform=MPI",
            available_route="serial/OpenMP build",
            alternative="rebuild with -DPOPS_USE_MPI=ON",
            source=source,
        ),
        _row(
            "supports_gpu",
            layout="uniform|amr",
            backend="production",
            platform="gpu",
            flags=flags,
            flag="supports_gpu",
            gpu=bool(_flag_value(flags, "supports_gpu")),
            limitation="GPU is available only for a Kokkos CUDA/HIP device build",
            requested="platform=GPU",
            available_route="host CPU platform",
            alternative=(
                "use the delivered host/serial route or a separately proved CUDA/HIP artifact"
            ),
            source=source,
        ),
        _row(
            "supports_stride",
            layout="uniform|amr",
            backend="production",
            platform="host",
            flags=flags,
            flag="supports_stride",
            limitation="real cell stride is carried only by the production/native route",
            requested="strided cell access",
            available_route="backend='production'",
            alternative="compile with backend='production'",
            source=source,
        ),
        _row(
            "supports_named_fields",
            layout="uniform|amr",
            backend="production",
            platform="host",
            flags=flags,
            flag="supports_named_fields",
            limitation="named aux-field transport",
            requested="named aux fields",
            available_route="native named-field transport",
            source=source,
        ),
        _row(
            "supports_partial_imex_mask",
            layout="uniform|amr",
            backend="production",
            platform="host",
            flags=flags,
            flag="supports_partial_imex_mask",
            limitation="no executable C++ Program primitive backs a partial IMEX mask",
            requested="partial IMEX mask",
            available_route="typed local implicit Program primitive where implemented",
            alternative=(
                "author an explicit typed Program; AMR currently has no implicit-source primitive"
            ),
            source=source,
        ),
        _row(
            "supports_custom_communicator",
            layout="uniform|amr",
            backend="none",
            platform="mpi",
            flags=flags,
            flag="supports_custom_communicator",
            limitation="no C++ route accepts a caller-provided MPI_Comm",
            requested="communicator != MPI_COMM_WORLD",
            available_route="MPI_COMM_WORLD or serial",
            alternative="use ExecutionContext.mpi_world() or a serial context",
            source=source,
        ),
    ]


def _inventory_rows(flags: Any, source: Any) -> list:
    mpi = bool(_flag_value(flags, "supports_mpi"))
    gpu = bool(_flag_value(flags, "supports_gpu"))
    from pops.mesh._amr._transfer_contracts import (
        CELL_CENTERED,
        NODE_CENTERED,
        ORIENTED_FACE_CENTERINGS,
    )

    physical_transfer_centerings = "/".join(
        (
            CELL_CENTERED.name,
            *(centering.name for centering in ORIENTED_FACE_CENTERINGS),
            NODE_CENTERED.name,
        )
    )
    return [
        _row(
            "layout:Uniform",
            layout="uniform",
            backend="module",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation=(
                "exact native-rank single-level Cartesian layout; polar is a rank-2 provider"
            ),
            source=source,
        ),
        _row(
            "layout:AMR",
            layout="amr",
            backend="production",
            platform="host",
            flags=flags,
            flag="supports_amr",
            mpi=mpi,
            gpu=gpu,
            limitation=(
                "resource-policy-controlled depth and hierarchy-selected transition ratios"
            ),
            requested="AMR hierarchy with an unauthenticated transition ratio",
            available_route="AMR hierarchy with explicit transition ratios",
            alternative="use Uniform or the native AMR envelope",
            source=source,
        ),
        _row(
            "spatial:finite_volume",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation="exact native-rank finite-volume production route",
            source=source,
        ),
        _row(
            "riemann:rusanov",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation="requires model max_wave_speed",
            source=source,
        ),
        _row(
            "riemann:hll",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation="requires physical_flux and wave_speeds",
            source=source,
        ),
        _row(
            "riemann:hllc",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation="requires exact HLLC model capability on the selected geometry",
            source=source,
        ),
        _row(
            "riemann:roe",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation="requires exact Roe dissipation capability on the selected geometry",
            source=source,
        ),
        # ADC-552: the typed wave-speed provider families a model can bind HLL to. Descriptor-level
        # (WaveSpeedProvider), so source is descriptor; the five signed families feed HLL, the
        # majorant family is the Rusanov spectral radius (HLL refuses it).
        _row(
            "wave_speeds:explicit_pair",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation="signed pair from typed Model.wave_speeds(...); HLL signed-wave source",
            source=source,
        ),
        _row(
            "wave_speeds:jacobian",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation="signed pair from flux-jacobian eigenvalues (m.wave_speeds_from_jacobian)",
            source=source,
        ),
        _row(
            "wave_speeds:pressure_derived",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation="signed pair from primitive 'p' + eigenvalues (historical path)",
            source=source,
        ),
        _row(
            "wave_speeds:einfeldt",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation="Einfeldt signed-speed estimate hook",
            source=source,
        ),
        _row(
            "wave_speeds:davis",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation="Davis signed-speed estimate hook",
            source=source,
        ),
        _row(
            "wave_speeds:max_wave_speed",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation="unsigned Rusanov majorant (spectral radius); refused by HLL, feeds Rusanov",
            source=source,
        ),
        _row(
            "reconstruction:firstorder",
            layout="uniform|amr",
            backend="production",
            limitation="ghost_depth=1",
            source=source,
        ),
        _row(
            "reconstruction:muscl",
            layout="uniform|amr",
            backend="production",
            limitation="ghost_depth=2; native limiters minmod/vanleer/mc/superbee",
            source=source,
        ),
        _row(
            "reconstruction:weno5",
            layout="uniform|amr",
            backend="production",
            limitation=(
                "ghost_depth=3; hierarchy-selected AMR in the compile-selected native rank "
                "selects the conservative order-5 "
                "cell-average provider from resolved spatial capabilities"
            ),
            source=source,
        ),
        _row(
            "limiter:mc",
            layout="uniform|amr",
            backend="production",
            limitation="native POPS_HD MC slope policy; formal_order=2; ghost_depth=2",
            source=source,
        ),
        _row(
            "limiter:superbee",
            layout="uniform|amr",
            backend="production",
            limitation="native POPS_HD Superbee slope policy; formal_order=2; ghost_depth=2",
            source=source,
        ),
        _row(
            "elliptic:geometric_mg",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation="native multigrid route; supports variable epsilon",
            source=source,
        ),
        _row(
            "elliptic:fft",
            layout="uniform",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation=(
                "exact Cartesian Dim in {1,2,3}, periodic, constant coefficient and canonical "
                "ordered MPI slabs; radix-2 fast path with diagnosed direct-DFT fallback"
            ),
            source=source,
        ),
        _row(
            "elliptic:mg_fac_defaults",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            status="partial",
            limitation=(
                "geometric MG/FAC defaults and debug diagnostics are still header-local; "
                "central SolverDefaults/logger follow-up is required"
            ),
            source=source,
        ),
        _row(
            "elliptic:fft_amr",
            layout="amr",
            backend="none",
            status="unavailable",
            limitation="FFT requires a single uniform periodic mesh, not AMR",
            requested="solver=FFT() with layout=AMR",
            available_route="GeometricMG() on AMR",
            alternative="use pops.solvers.elliptic.GeometricMG()",
            source=source,
        ),
        _row(
            "mesh:nd_storage_arithmetic",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            status="available",
            limitation=(
                "one compile-time-ranked Index/Box/Fab/MultiFab arithmetic core; the "
                "resolved artifact retains exactly one dimension in {1,2,3}"
            ),
            source=source,
        ),
        _row(
            "amr:refinement_ratio",
            layout="amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            status="partial",
            limitation=(
                "AMR transition ratios are exact hierarchy properties; the native runtime does "
                "not advertise a process-global ratio invariant"
            ),
            source=source,
        ),
        _row(
            "amr:transition_envelope",
            layout="amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            status="partial",
            limitation=(
                "transitions are 2D isotropic ratio-(2,2); every transition must share "
                "one isotropic buffer and one lookahead value"
            ),
            source=source,
        ),
        _row(
            "amr:hierarchy_policy_routes",
            layout="amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            status="partial",
            limitation=(
                "native hierarchy route is shared_n_level with berger_rigoutsos "
                "clustering, box_array patch generation, and the resolved prepared "
                "load-balance provider (space-filling curve by default)"
            ),
            source=source,
        ),
        _row(
            "amr:transfer_contracts",
            layout="amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            status="partial",
            limitation=(
                "physical transfer routes are exact dense %s "
                "contracts; cell-centered state provides restriction, coarse-fine fill and "
                "qualified temporal interpolation; complete oriented Cartesian face vectors "
                "provide divergence-preserving prolongation and primitive node fields provide "
                "multilinear prolongation; derived fields recompute through elliptic_solve and "
                "caches rebuild through patch_topology"
            )
            % physical_transfer_centerings,
            source=source,
        ),
        _row(
            "parallel:mpi_world_communicator",
            layout="uniform|amr",
            backend="production",
            platform="mpi",
            mpi=mpi,
            status="available" if mpi else "unavailable",
            limitation=(
                "exact MPI_COMM_WORLD execution is proved by the native module and ExecutionContext"
                if mpi
                else "this native module was not built with POPS_USE_MPI=ON"
            ),
            requested="ExecutionContext.mpi_world()",
            available_route=("ExecutionContext.mpi_world()" if mpi else "serial ExecutionContext"),
            alternative=None if mpi else "rebuild with -DPOPS_USE_MPI=ON",
            source=source,
        ),
        _row(
            "parallel:custom_communicator",
            layout="uniform|amr",
            backend="none",
            platform="mpi",
            mpi=mpi,
            status="unavailable",
            limitation="no native route accepts a caller-provided MPI_Comm",
            requested="communicator != MPI_COMM_WORLD",
            available_route="MPI_COMM_WORLD or serial",
            alternative="use ExecutionContext.mpi_world() or a serial context",
            source=source,
        ),
        _row(
            "precision:single_or_mixed",
            layout="uniform|amr",
            backend="none",
            platform="host",
            status="unavailable",
            limitation=(
                "PrecisionPolicy is representable, but the native providers currently "
                "instantiate pops::Real as binary64 only"
            ),
            requested="precision=single or precision=mixed",
            available_route="precision=double",
            alternative="use double precision or implement a non-binary64 native provider",
            source=source,
        ),
        _row(
            "runtime:kokkos_lifecycle",
            layout="uniform|amr",
            backend="production",
            platform="host|gpu",
            mpi=mpi,
            gpu=gpu,
            status="partial",
            limitation=(
                "Kokkos is lazily initialized by PoPS on first allocation/kernel unless "
                "the caller already initialized it; runtime_environment_report() exposes "
                "ownership and initialized/finalized state"
            ),
            source=source,
        ),
        _row(
            "runtime:allocator_lifetime",
            layout="uniform|amr",
            backend="production",
            platform="host|gpu",
            mpi=mpi,
            gpu=gpu,
            status="partial",
            limitation=(
                "Kokkos builds use a process-lifetime ManagedArena; blocks are released "
                "by a Kokkos finalize hook and the arena tables intentionally survive "
                "process teardown"
            ),
            source=source,
        ),
        _row(
            "krylov:cg_bicgstab_gmres_richardson",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation="matrix-free Krylov over native MultiFab primitives",
            source=source,
        ),
        _row(
            "program:hierarchy_scoped_solve",
            layout="uniform|amr",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            status="partial",
            limitation=(
                "Program.solve and its provider protocol are physics-independent; AMR "
                "hierarchy lowering currently supports one top-level linear solve with "
                "CompositeTensorFAC()"
            ),
            source=source,
        ),
        _row(
            "program_context:system",
            layout="uniform",
            backend="production",
            platform="host",
            mpi=mpi,
            gpu=gpu,
            limitation="compiled ProgramContext install on System",
            source=source,
        ),
        _row(
            "program_context:amr",
            layout="amr",
            backend="production",
            platform="host",
            flags=flags,
            flag="supports_amr",
            mpi=mpi,
            gpu=gpu,
            limitation="AMR program install requires target='amr_system'",
            source=source,
        ),
        _row(
            "output:scientific_v1",
            layout="uniform|amr",
            backend="runtime",
            platform="host|mpi",
            mpi=mpi,
            limitation=(
                "typed SERIAL/ROOT/COLLECTIVE/PER_RANK publication; each format "
                "advertises its exact supported modes"
            ),
            source=source,
        ),
        _row(
            "checkpoint:uniform_accepted_state_v5",
            layout="uniform",
            backend="runtime",
            platform="host|mpi",
            mpi=mpi,
            limitation="single-file strict accepted-state checkpoint",
            source=source,
        ),
        _row(
            "checkpoint:amr_accepted_state_v7",
            layout="amr",
            backend="runtime",
            platform="host|mpi",
            mpi=mpi,
            limitation=(
                "strict accepted-state checkpoint includes the runtime-owned AMR tagging "
                "payload and accepted shared-interface flux audit; MPI_COMM_WORLD uses one "
                "rank-0 publication with collective capture and consensus"
            ),
            source=source,
        ),
        _row(
            "checkpoint:parallel_hdf5",
            layout="uniform|amr",
            backend="none",
            platform="mpi",
            status="unavailable",
            limitation="parallel HDF5 checkpoint is not a native checkpoint route",
            requested="restartable checkpoint encoded as parallel HDF5",
            available_route="strict accepted-state NPZ checkpoint (uniform v6, AMR v9)",
            alternative="use RuntimeInstance.checkpoint() or the typed Checkpoint consumer",
            source=source,
        ),
        _row(
            "checkpoint:amr_dynamic_regrid",
            layout="amr",
            backend="runtime",
            platform="host",
            flags=flags,
            flag="supports_amr",
            mpi=mpi,
            limitation=(
                "strict v7 accepted-state restart; exact rank-local AMR ownership and "
                "compiled-Program publications keep the native rank count"
            ),
            source=source,
        ),
    ] + _python_contract_rows(flags, source)


def native_capability_matrix(
    *,
    owner: str = "module",
    layout: str = "module",
    target: str = "module",
    flags: Any = None,
    source: Any = None,
) -> Any:
    """Return the ADC-549 native route matrix.

    ``flags`` can be supplied by a compiled artifact manifest. When absent, the built module's
    C++ ``module_capabilities(target)`` is used. The returned rows always expose:
    feature, layout, backend, platform, MPI, GPU, status, limitation and error_message.
    """
    report = native_capability_report(target, flags=flags, source=source)
    return CapabilityRouteMatrix(
        owner,
        layout,
        report.routes,
        schema_version=report.schema_version,
        abi_version=report.abi_version,
        target=report.target,
        abi_key=report.abi_key,
        platform=report.platform,
    )
