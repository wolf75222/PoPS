"""pops.codegen.inspect_compiled -- INERT introspection of a compiled artifact (Spec 5 sec.12).

The compiled-artifact introspection surface (criteria #44-49, epic ADC-479): value classes and pure
builders populated from metadata already carried by :class:`pops.codegen.loader.CompiledProblem`
(its lowered ``pops.time.Program`` and physical model), plus the compile artifacts on disk.

  - :class:`Arguments` (sec.12.2, #44-45) lists the RUNTIME inputs expected by
    :func:`pops.bind`, without binding or reading a runtime array.
  - :class:`MemoryEstimate` (sec.12.3, #46) turns the Program's GRID-RELATIVE static cost
    (``Program.estimate``: field-sized passes) into an ABSOLUTE byte estimate over a mesh shape,
    as a FORMULA -- it allocates nothing (no ``MultiFab``). Every assumption is inspectable.
  - metadata attributes live on :class:`CompiledProblem`; helpers feed its report methods.

Nothing here compiles, binds, dlopens or allocates: the builders read Python-side metadata only.
The module imports ``pops.mesh`` lazily (in-function) to respect the codegen layering (a codegen
module may not import ``pops.mesh`` at module scope; cf. tests/python/architecture/test_import_graph.py).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
from typing import Any

from pops.codegen._artifact_models import (
    aggregate_model_metadata as _model_metadata,
    artifact_model_metadata as _artifact_model_metadata,
)
from pops._report import Report


class MemoryEstimateCapabilityError(ValueError):
    """The requested absolute estimate lacks a required, authoritative capability fact."""

    def __init__(self, message: str, *, field: str, actual: Any = None) -> None:
        super().__init__(message)
        self.field = field
        self.actual = actual


@dataclass(frozen=True, slots=True)
class _MemoryRuntimeContext:
    """Validated native facts required to turn a structural formula into bytes."""

    dimension: int
    real_bytes: int
    amr_refinement_ratio: int


@dataclass(frozen=True, slots=True)
class _MemoryLayoutContext:
    """Validated layout facts used by the hierarchy part of an absolute estimate."""

    kind: str
    dimension: int
    max_levels: int
    ratio: int | None


def _native_memory_context() -> _MemoryRuntimeContext:
    """Read the native precision and AMR facts; an absolute estimate has no source-only mode."""
    mod = None
    for name in ("_pops", "pops._pops"):
        try:
            mod = importlib.import_module(name)
            break
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    if mod is None:
        raise MemoryEstimateCapabilityError(
            "estimate_memory requires _pops.runtime_environment_report(): absolute byte precision is "
            "unknown in a source-only installation", field="runtime.precision")
    fn = getattr(mod, "runtime_environment_report", None)
    if not callable(fn):
        raise MemoryEstimateCapabilityError(
            "estimate_memory requires callable _pops.runtime_environment_report()",
            field="runtime_environment_report")
    try:
        raw_report = fn()
        if not isinstance(raw_report, Mapping):
            raise TypeError("runtime environment report is not a mapping")
        report = dict(raw_report)
    except Exception as exc:
        raise MemoryEstimateCapabilityError(
            "_pops.runtime_environment_report() failed or returned a malformed mapping",
            field="runtime_environment_report") from exc

    values = {}
    for key in ("dimension", "real_bytes", "amr_refinement_ratio"):
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MemoryEstimateCapabilityError(
                "estimate_memory requires native runtime_environment_report[%r] as a positive int "
                "(got %r)" % (key, value), field="runtime.%s" % key, actual=value)
        values[key] = value
    if values["dimension"] != 2:
        raise MemoryEstimateCapabilityError(
            "estimate_memory currently implements an explicit 2D perimeter/hierarchy formula; native "
            "dimension=%d requires a dimension-aware estimator" % values["dimension"],
            field="runtime.dimension", actual=values["dimension"])
    return _MemoryRuntimeContext(**values)


# ---------------------------------------------------------------------------
# sec.12.2 -- Arguments: the runtime inputs the artifact expects at bind
# ---------------------------------------------------------------------------

class Arguments(Report):
    """The concrete runtime values a compiled artifact expects at :func:`pops.bind`.

    A plain, inert value describing what a caller must SUPPLY to bind the artifact -- distinct
    from ``CompiledProblem.requirements`` (the compile-time constraints). It lists, per group:

      - ``instances``: the physics blocks the Program commits (name -> state space / component
        count / required), supplied through ``initial_state=``;
      - ``params``: the model's declared parameters (name -> type / kind / required), supplied
        through ``params=`` (only ``kind == "runtime"`` is settable at bind);
      - ``aux``: the static external aux inputs the model declares (name -> layout / required),
        supplied through ``aux=``; channels produced by a resolved field provider are excluded;
      - ``outputs``: histories and diagnostics the Program records (informational); resolved field
        provider outputs remain inspectable on the compiled plan and are never duplicated as inputs;
      - ``layout_runtime``: the mesh layout the artifact targets (layout / requires_mpi /
        ghost_depth).

    Resolved field discretizations and their solver providers are compile-time plan evidence, not
    bind inputs. They remain available from the resolved plan and compiled-artifact inspection and
    are deliberately absent here.

    It is built by :func:`build_arguments` from the carried Program + model; it neither compiles,
    binds nor reads any runtime array. ``str(args)`` is a readable table; :meth:`to_dict` /
    :meth:`to_json` serialise it. It uses the shared internal report base; its
    ``to_dict`` keeps the historical shape (no ``report_type`` stamp) so a consumer is unchanged.
    """

    report_type = "arguments"
    schema_version = 2

    def __init__(self, *, instances: Any, params: Any, aux: Any, outputs: Any,
                 layout_runtime: Any, program_name: Any = None) -> None:
        self.instances = dict(instances)
        self.params = dict(params)
        self.aux = dict(aux)
        self.outputs = dict(outputs)
        self.layout_runtime = dict(layout_runtime)
        self.program_name = program_name

    def to_dict(self) -> dict:
        """A plain-dict view of every argument group (JSON-ready)."""
        return {"program": self.program_name,
                "instances": {k: dict(v) for k, v in self.instances.items()},
                "params": {k: dict(v) for k, v in self.params.items()},
                "aux": {k: dict(v) for k, v in self.aux.items()},
                "outputs": {k: dict(v) for k, v in self.outputs.items()},
                "layout_runtime": dict(self.layout_runtime)}

    def __str__(self) -> str:
        lines = ["arguments for compiled artifact %r (bind inputs)"
                 % (self.program_name or "problem")]
        lines.append("  instances (bind initial_state=):")
        for name, spec in sorted(self.instances.items()):
            lines.append("    %-14s state=%s comps=%s required=%s"
                         % (name, spec.get("state"), spec.get("components"),
                            spec.get("required")))
        lines.append("  params (bind params=):")
        for name, spec in sorted(self.params.items()):
            lines.append("    %-14s type=%s kind=%s required=%s"
                         % (name, spec.get("type"), spec.get("kind"), spec.get("required")))
        lines.append("  aux (bind aux=):")
        for name, spec in sorted(self.aux.items()):
            lines.append("    %-14s layout=%s required=%s"
                         % (name, spec.get("layout"), spec.get("required")))
        lines.append("  outputs (resolved artifact metadata, not bind inputs):")
        for name, spec in sorted(self.outputs.items()):
            lines.append("    %-14s kind=%s" % (name, spec.get("kind")))
        lr = self.layout_runtime
        lines.append("  resolved runtime target : layout=%s requires_mpi=%s ghost_depth=%s"
                     % (lr.get("layout"), lr.get("requires_mpi"), lr.get("ghost_depth")))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return ("Arguments(instances=%d, params=%d, aux=%d)"
                % (len(self.instances), len(self.params), len(self.aux)))


def build_arguments(compiled: Any) -> Arguments:
    """Build the :class:`Arguments` of a compiled artifact from its carried metadata (sec.12.2).

    Sources, all Python-side (no compile / bind / runtime read):

      - instances: the blocks the Program COMMITS (``program.commits()`` -- the blocks it advances);
        each is required and carries the model's conservative state space + component count;
      - params: the model's declared parameters (``model.params``); ``kind`` is the declared kind
        (``runtime`` settable at bind, ``const`` frozen at compile);
      - aux: the model's named external aux inputs (``model.aux_extra_names`` minus the exact,
        owner-scoped components produced by resolved field plans), each required;
      - outputs: the values the Program records for output (``store_history`` / ``record`` ops);
      - layout_runtime: every exact compiled layout partition, its target, MPI optionality and
        per-block ghost depth. A multi-layout artifact is reported as an aggregate of qualified
        partitions, never through a representative layout.
    """
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact

    if type(compiled) is not CompiledSimulationArtifact:
        raise TypeError("build_arguments requires a CompiledSimulationArtifact")
    model_rows = _artifact_model_metadata(compiled)
    if len(compiled.layout_programs) == 1:
        row = compiled.layout_programs[0]
        return _build_arguments(
            compiled, getattr(row.program, "program", None), model_rows,
            program_name=getattr(row.program, "program_name", None))
    if not compiled.layout_programs:
        return _build_arguments(compiled, None, model_rows)

    partitions = tuple(
        (row.layout_id, build_layout_arguments(compiled, row.layout_id))
        for row in compiled.layout_programs
    )
    instances = {}
    aux = {}
    outputs = {}
    ghost_depth_by_block = {}
    layout_rows = {}
    for layout_id, arguments in partitions:
        overlap = set(instances) & set(arguments.instances)
        if overlap:
            raise ValueError("layout Program partitions share block(s) %s" % sorted(overlap))
        instances.update(arguments.instances)
        for target, source in ((aux, arguments.aux), (outputs, arguments.outputs)):
            target.update({"%s::%s" % (layout_id, name): value
                           for name, value in source.items()})
        depths = dict(arguments.layout_runtime.get("ghost_depth_by_block") or {})
        ghost_depth_by_block.update(depths)
        layout_rows[layout_id] = {
            "blocks": sorted(arguments.instances),
            "target": compiled.program_for_layout(layout_id).target,
            "ghost_depth_by_block": depths,
        }
    from ._inspect_params import build_parameter_arguments

    params = build_parameter_arguments(compiled, _merge_parameter_metadata(model_rows))
    return Arguments(
        instances=instances,
        params=params,
        aux=aux,
        outputs=outputs,
        layout_runtime={
            "layout": "multi",
            "requires_mpi": False,
            "requires_gpu": False,
            "ghost_depth": None,
            "ghost_depth_by_block": ghost_depth_by_block,
            "layouts": layout_rows,
        },
        program_name={layout_id: arguments.program_name
                      for layout_id, arguments in partitions},
    )


def build_layout_arguments(compiled: Any, layout_id: str) -> Arguments:
    """Project one exact layout partition without using an aggregate representative."""
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact

    if type(compiled) is not CompiledSimulationArtifact:
        raise TypeError("build_layout_arguments requires a CompiledSimulationArtifact")
    layout_program = compiled.program_for_layout(layout_id)
    names = frozenset(layout_program.block_names)
    model_rows = tuple(
        row for row in _artifact_model_metadata(compiled) if row.block_name in names)
    if {row.block_name for row in model_rows} != names:
        raise ValueError("layout Program has no exact compiled-model partition")
    return _build_arguments(
        compiled,
        getattr(layout_program.program, "program", None),
        model_rows,
        program_name=getattr(layout_program.program, "program_name", None),
    )


def build_component_arguments(compiled: Any) -> Arguments:
    """Advanced low-level counterpart for exact compiled component handles."""
    from pops.codegen._artifact_models import component_model_metadata
    from pops.codegen.loader import CompiledModel, CompiledProblem

    if type(compiled) not in (CompiledModel, CompiledProblem):
        raise TypeError("build_component_arguments requires an exact compiled component")
    program = compiled.program if type(compiled) is CompiledProblem else None
    return _build_arguments(compiled, program, component_model_metadata(compiled))


def _merge_parameter_metadata(model_rows: Any) -> dict[str, Any]:
    """Merge local parameter declarations only after proving homonyms identical."""
    params = {}
    owners = {}
    for row in model_rows:
        for name, value in row.params.items():
            if name in params and params[name] != value:
                raise ValueError(
                    "compiled blocks %r and %r declare conflicting parameter metadata for %r"
                    % (owners[name], row.block_name, name))
            params[name] = value
            owners.setdefault(name, row.block_name)
    return params


def _field_output_components_by_block(compiled: Any) -> dict[str, tuple[str, ...]]:
    """Return exact native field-output components, qualified by their owning block.

    A ``GradientOutput`` label is semantic metadata and may lower to two differently named aux
    channels.  The authenticated ``output_route`` is the one authority shared by System and AMR
    installers.  Low-level component handles carry no total resolved plan and therefore return an
    empty mapping; a malformed total plan fails loudly instead of falling back to output labels.
    """
    plan = getattr(compiled, "plan", None)
    if plan is None:
        return {}
    field_plans = getattr(plan, "field_plans", None)
    if field_plans is None:
        return {}
    if not isinstance(field_plans, Mapping):
        raise TypeError("compiled field plans must be an exact mapping")

    produced: dict[str, list[str]] = {}
    for name, field_plan in field_plans.items():
        native_options = getattr(field_plan, "native_options", None)
        route = native_options.get("output_route") if isinstance(native_options, Mapping) else None
        owner = route.get("owner_block") if isinstance(route, Mapping) else None
        components = route.get("components") if isinstance(route, Mapping) else None
        if (not isinstance(owner, str) or not owner
                or not isinstance(components, (tuple, list)) or not components
                or any(not isinstance(component, str) or not component
                       for component in components)):
            raise TypeError(
                "compiled field plan %r has no exact owner-scoped output route" % name)
        bucket = produced.setdefault(owner, [])
        overlap = set(bucket).intersection(components)
        if overlap:
            raise ValueError(
                "compiled field plans have competing output components on block %r: %s"
                % (owner, sorted(overlap)))
        bucket.extend(components)
    return {owner: tuple(components) for owner, components in produced.items()}


def _build_aux_arguments(model_rows: Any, produced_by_block: Any) -> dict[str, dict[str, Any]]:
    """Build external aux inputs without letting one block exempt another block's homonym."""
    aux_args: dict[str, dict[str, Any]] = {}
    for row in model_rows:
        produced = frozenset(produced_by_block.get(row.block_name, ()))
        for name in row.aux_names:
            if name not in produced:
                aux_args.setdefault(name, {"layout": "cell", "required": True})
    return aux_args


def _build_arguments(
    compiled: Any,
    program: Any,
    model_rows: Any,
    *,
    program_name: Any = None,
) -> Arguments:
    params = _merge_parameter_metadata(model_rows)

    # Instances: the blocks the Program commits. A read-only block (never committed) is still a
    # bind input, but the Program only references blocks it commits or reads; the commit set is the
    # authoritative list of advanced blocks (criterion 23: the block is bound by name).
    commits = {}
    if program is not None and hasattr(program, "commits"):
        commits = program.commits()
    instances = {}
    from pops.time.references import block_name as _block_name, handle_data
    by_block = {row.block_name: row for row in model_rows if row.block_name is not None}
    for state_ref in sorted(commits, key=lambda item: item.qualified_id):
        name = _block_name(state_ref.block_ref)
        row = by_block.get(name)
        if row is None:
            raise ValueError(
                "compiled Program references block %r without exact model metadata" % name
            )
        instances[name] = {
            "state": row.state_space,
            "components": row.n_vars,
            "required": True,
            "conservative": list(row.cons_names),
            "block_identity": handle_data(state_ref.block_ref),
            "state_identity": handle_data(state_ref),
        }
    if not instances:
        # AMR without a whole-system Program advances every native InstallPlan block. The plan, not
        # the first returned CompiledModel, is therefore the complete instance authority.
        for row in model_rows:
            name = row.block_name or "block"
            instances[name] = {
                "state": row.state_space,
                "components": row.n_vars,
                "required": True,
                "conservative": list(row.cons_names),
            }

    from ._inspect_params import build_parameter_arguments
    param_args = build_parameter_arguments(compiled, params)

    produced_by_block = _field_output_components_by_block(compiled)
    aux_args = _build_aux_arguments(model_rows, produced_by_block)

    outputs = {}
    if program is not None:
        for value in getattr(program, "_values", []):
            if value.op == "store_history":
                outputs[value.name or "history"] = {"kind": "history"}
            elif value.op == "record" or value.op == "record_scalar":
                outputs[value.name or "diagnostic"] = {"kind": "diagnostic"}

    ghost_depth_by_block = _ghost_depth_by_block(compiled, tuple(instances))
    # The scalar remains an exact conservative aggregate for legacy/native consumers; each block's
    # authoritative value is retained separately and may differ.
    ghost_depth = max(ghost_depth_by_block.values())
    # The runtime LAYOUT the artifact targets: "amr" for an AMR-route CompiledModel (target=
    # 'amr_system', ADC-515) so ``arguments()`` reports the native per-block AMR loader; a whole-system
    # Program handle stays "system" (its only target today).
    _amr = getattr(compiled, "target", "system") == "amr_system"
    layout_kind = "amr" if _amr else "system"
    mpi_values = [
        bool(row.model.caps["mpi"])
        for row in model_rows
        if getattr(row.model, "caps", None) and "mpi" in row.model.caps
    ]
    supports_mpi = bool(mpi_values) and len(mpi_values) == len(model_rows) and all(mpi_values)
    layout_runtime = {"layout": layout_kind, "requires_mpi": False, "requires_gpu": False,
                      "ghost_depth": ghost_depth, "ghost_depth_by_block": ghost_depth_by_block,
                      "supports_mpi": supports_mpi}

    return Arguments(instances=instances, params=param_args, aux=aux_args,
                     outputs=outputs, layout_runtime=layout_runtime,
                     program_name=(getattr(compiled, "program_name", None)
                                   if program_name is None else program_name))


def _ghost_depth(compiled: Any) -> int:
    """Return the exact maximum halo depth derived from every resolved block plan."""
    plan = getattr(compiled, "plan", None)
    blocks = getattr(plan, "blocks", None)
    if not isinstance(blocks, tuple) or not blocks:
        raise ValueError(
            "compiled artifact has no resolved block plan from which to derive ghost depth")
    return max(_ghost_depth_by_block(compiled, tuple(block.name for block in blocks)).values())


def _ghost_depth_by_block(compiled: Any, block_names: Any) -> dict[str, int]:
    """Derive per-block halo depths; never substitute a reconstruction default."""
    plan = getattr(compiled, "plan", None)
    blocks = getattr(plan, "blocks", None)
    if not isinstance(blocks, tuple):
        raise ValueError(
            "compiled artifact has no resolved block plan from which to derive ghost depth")
    by_name = {block.name: block for block in blocks}
    names = tuple(block_names)
    if not names:
        raise ValueError("ghost depth derivation requires at least one compiled block")
    if len(names) != len(set(names)) or set(names) - set(by_name):
        raise ValueError("ghost depth derivation received non-exact compiled block names")
    result = {}
    for name in names:
        block = by_name[name]
        candidates = []
        numerics = getattr(block, "numerics", None)
        if numerics is not None:
            rates = getattr(numerics, "rates", None)
            if not isinstance(rates, tuple) or not rates:
                raise ValueError(
                    "compiled block %r numerics carries no rate stencil ghost depth" % name)
            candidates.append(max(row.method.ghost_depth for row in rates))
        spatial = getattr(block, "spatial", None)
        if spatial is not None:
            value = spatial.get("ghost_depth") if isinstance(spatial, Mapping) \
                else getattr(spatial, "ghost_depth", None)
            if value is not None:
                candidates.append(value)
        if not candidates:
            raise ValueError(
                "compiled block %r has no exact ghost depth in its resolved numerics/spatial plan"
                % name)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in candidates):
            raise TypeError("compiled block %r ghost depth must be an integer >= 1" % name)
        if any(value != candidates[0] for value in candidates[1:]):
            raise ValueError(
                "compiled block %r carries conflicting resolved ghost depths %s"
                % (name, candidates))
        result[name] = candidates[0]
    return result


# ---------------------------------------------------------------------------
# sec.12.3 -- MemoryEstimate: an absolute byte FORMULA over a mesh shape
# ---------------------------------------------------------------------------

class MemoryEstimate(Report):
    """An ABSOLUTE memory estimate for a compiled artifact on a given mesh (Spec 5 sec.12.3).

    A FORMULA, not an allocation: it multiplies the Program's grid-relative static cost
    (``Program.estimate``: field-sized buffer passes, scratch buffer count) by the cell count and
    the per-cell byte size, and adds the persistent state / field-output / aux footprint. It never
    constructs a ``MultiFab``. Every figure is a category in :attr:`categories` (bytes); the
    :attr:`assumptions` list records what the estimate takes for granted (it is CONSERVATIVE: it
    over-counts scratch as if no codegen reuse happened beyond the static reuse report, and ignores
    in-solver V-cycle traffic). :meth:`by_block` / :meth:`by_solver` / :meth:`by_scratch` slice it.
    It uses the internal report base; ``to_dict`` keeps the established shape.
    """

    report_type = "memory_estimate"
    schema_version = 1

    def __init__(self, *, categories: Any, cells: Any, mesh_shape: Any, n_cons: Any, n_aux: Any,
                 scratch_buffers: Any, assumptions: Any, conservative: Any = True,
                 layout: Any = "system") -> None:
        self.categories = dict(categories)   # category -> bytes
        self.cells = int(cells)
        self.mesh_shape = tuple(mesh_shape)
        self.n_cons = int(n_cons)
        self.n_aux = int(n_aux)
        self.scratch_buffers = int(scratch_buffers)
        self.assumptions = list(assumptions)
        self.conservative = bool(conservative)
        self.layout = str(layout)

    @property
    def total_bytes(self) -> Any:
        """Sum of every category, in bytes."""
        return sum(self.categories.values())

    def by_block(self) -> dict:
        """The per-block (state-sized) categories: persistent state, RHS / state scratch."""
        keys = ("state", "rhs_scratch", "state_scratch", "field_output", "aux")
        return {k: self.categories[k] for k in keys if k in self.categories}

    def by_solver(self) -> dict:
        """The elliptic / Krylov / multigrid categories (the field solves)."""
        keys = ("scalar_field", "krylov", "multigrid")
        return {k: self.categories[k] for k in keys if k in self.categories}

    def by_scratch(self) -> dict:
        """The transient scratch categories (RHS / state scratch, halo, MPI buffers)."""
        keys = ("rhs_scratch", "state_scratch", "halo", "mpi_buffer", "amr_patch")
        return {k: self.categories[k] for k in keys if k in self.categories}

    def to_dict(self) -> dict:
        """A plain-dict view: every category, the total, the mesh + assumptions (JSON-ready)."""
        return {"total_bytes": self.total_bytes, "categories": dict(self.categories),
                "cells": self.cells, "mesh_shape": list(self.mesh_shape),
                "n_cons": self.n_cons, "n_aux": self.n_aux,
                "scratch_buffers": self.scratch_buffers, "layout": self.layout,
                "conservative": self.conservative, "assumptions": list(self.assumptions)}

    def _mib(self, n_bytes: Any) -> float:
        return n_bytes / (1024.0 * 1024.0)

    def __str__(self) -> str:
        lines = ["memory estimate on mesh %s (%d cells, %d cons, %d aux) -- %s formula"
                 % (self.mesh_shape, self.cells, self.n_cons, self.n_aux,
                    "conservative" if self.conservative else "tight")]
        for name in sorted(self.categories):
            lines.append("  %-14s %12d B  (%8.2f MiB)"
                         % (name, self.categories[name], self._mib(self.categories[name])))
        lines.append("  %-14s %12d B  (%8.2f MiB)"
                     % ("TOTAL", self.total_bytes, self._mib(self.total_bytes)))
        if self.assumptions:
            lines.append("  assumptions:")
            for note in self.assumptions:
                lines.append("    - %s" % note)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return ("MemoryEstimate(total=%d B, cells=%d, categories=%d)"
                % (self.total_bytes, self.cells, len(self.categories)))


def _capability_data(provider: Any, *, where: str) -> dict[str, Any]:
    """Read the small typed capability protocol without coupling to descriptor classes."""
    capabilities = getattr(provider, "capabilities", None)
    if not callable(capabilities):
        raise MemoryEstimateCapabilityError(
            "estimate_memory requires %s.capabilities()" % where,
            field="%s.capabilities" % where, actual=type(provider).__name__)
    reported = capabilities()
    to_dict = getattr(reported, "to_dict", None)
    if not callable(to_dict):
        raise MemoryEstimateCapabilityError(
            "estimate_memory requires %s.capabilities() to return a typed value with to_dict()"
            % where, field="%s.capabilities" % where, actual=type(reported).__name__)
    data = to_dict()
    if not isinstance(data, Mapping):
        raise MemoryEstimateCapabilityError(
            "estimate_memory requires %s.capabilities().to_dict() to return a mapping" % where,
            field="%s.capabilities" % where, actual=type(data).__name__)
    return dict(data)


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MemoryEstimateCapabilityError(
            "estimate_memory requires %s as a positive integer (got %r)" % (field, value),
            field=field, actual=value)
    return value


def _mesh_shape(mesh: Any, context: _MemoryRuntimeContext) -> tuple:
    """Read extents from the sole public Cartesian-grid capability-and-data protocol."""
    from pops.mesh.grid import CartesianGrid

    if type(mesh) is not CartesianGrid:
        raise MemoryEstimateCapabilityError(
            "estimate_memory requires an exact pops.mesh.CartesianGrid over a framed Rectangle "
            "(got %r)" % type(mesh).__name__, field="mesh", actual=type(mesh).__name__)
    capabilities = _capability_data(mesh, where="mesh")
    if capabilities.get("geometry") != "cartesian":
        raise MemoryEstimateCapabilityError(
            "estimate_memory requires a Cartesian mesh/grid capability (got geometry=%r)"
            % capabilities.get("geometry"), field="mesh.geometry",
            actual=capabilities.get("geometry"))
    dimension = _positive_int(capabilities.get("dim"), field="mesh.dimension")
    if dimension != context.dimension:
        raise MemoryEstimateCapabilityError(
            "estimate_memory mesh dimension=%r disagrees with native dimension=%r"
            % (dimension, context.dimension), field="mesh.dimension", actual=dimension)

    cells = mesh.cells
    if not isinstance(cells, tuple) or len(cells) != dimension:
        raise MemoryEstimateCapabilityError(
            "estimate_memory requires mesh.cells with exactly %d extents (got %r)"
            % (dimension, cells), field="mesh.cells", actual=cells)
    shape = tuple(_positive_int(value, field="mesh.cells[%d]" % index)
                  for index, value in enumerate(cells))
    if dimension != 2:
        raise MemoryEstimateCapabilityError(
            "estimate_memory currently implements a 2D perimeter/hierarchy formula; mesh dimension=%d "
            "requires a dimension-aware estimator" % dimension,
            field="mesh.dimension", actual=dimension)
    return shape[0] * shape[1], shape


def _layout_dimension(layout: Any, capabilities: dict[str, Any]) -> int:
    """Read a layout's explicit dimension, or the capability data of its public grid provider."""
    dimension = capabilities.get("dim")
    if dimension is None:
        for name in ("grid", "mesh", "base"):
            provider = getattr(layout, name, None)
            if provider is not None:
                dimension = _capability_data(provider, where="layout.%s" % name).get("dim")
                break
    return _positive_int(dimension, field="layout.dimension")


def _layout_context(layout: Any, context: _MemoryRuntimeContext) -> _MemoryLayoutContext:
    """Validate the class-independent layout capability protocol used by memory estimation."""
    capabilities = _capability_data(layout, where="layout")
    kind = capabilities.get("layout")
    if kind not in ("uniform", "amr"):
        raise MemoryEstimateCapabilityError(
            "estimate_memory supports only layout='uniform' or layout='amr' (got %r)" % kind,
            field="layout.kind", actual=kind)
    dimension = _layout_dimension(layout, capabilities)
    if dimension != context.dimension:
        raise MemoryEstimateCapabilityError(
            "estimate_memory layout dimension=%r disagrees with native dimension=%r"
            % (dimension, context.dimension), field="layout.dimension", actual=dimension)

    max_levels = _positive_int(
        capabilities.get("max_levels", capabilities.get("levels")), field="layout.max_levels")
    if kind == "uniform":
        if max_levels != 1 or capabilities.get("supports_amr") is not False:
            raise MemoryEstimateCapabilityError(
                "estimate_memory uniform layout must explicitly report max_levels=1 and supports_amr=False",
                field="layout.max_levels", actual=max_levels)
        return _MemoryLayoutContext(kind, dimension, max_levels, None)

    ratios = capabilities.get("transition_ratios")
    if not isinstance(ratios, (tuple, list)) or len(ratios) != max_levels - 1:
        raise MemoryEstimateCapabilityError(
            "estimate_memory AMR layout requires transition_ratios for every level transition",
            field="layout.transition_ratios", actual=ratios)
    normalized_ratios = tuple(
        _positive_int(value, field="layout.transition_ratios[%d]" % index)
        for index, value in enumerate(ratios))
    if max_levels == 1:
        return _MemoryLayoutContext(kind, dimension, max_levels, None)
    declared_ratio = capabilities.get("ratio")
    ratio = (normalized_ratios[0] if declared_ratio is None
             else _positive_int(declared_ratio, field="layout.ratio"))
    if any(value != ratio for value in normalized_ratios):
        raise MemoryEstimateCapabilityError(
            "estimate_memory requires one normalized AMR ratio; transition_ratios=%r disagree with ratio=%d"
            % (normalized_ratios, ratio), field="layout.transition_ratios", actual=normalized_ratios)
    if ratio != context.amr_refinement_ratio:
        raise MemoryEstimateCapabilityError(
            "estimate_memory AMR ratio=%d disagrees with native ratio=%d" % (
                ratio, context.amr_refinement_ratio), field="layout.ratio", actual=ratio)
    return _MemoryLayoutContext(kind, dimension, max_levels, ratio)


def build_memory_estimate(compiled: Any, mesh: Any, *, platform: Any = None,
                          layout: Any = None) -> MemoryEstimate:
    """Build the :class:`MemoryEstimate` for a compiled artifact on ``mesh`` (sec.12.3).

    A pure FORMULA over the Program's static cost (``Program.estimate``) and the carried model's
    component counts -- it allocates nothing. ``B`` is read from the loaded native runtime report;
    an absolute estimate is rejected rather than guessing precision in a source-only install.

      - ``state``        = n_cons * C * B           (the persistent conservative state)
      - ``field_output`` = (#field solves) * C * B  (one scalar field per elliptic solve)
      - ``aux``          = n_aux * C * B            (the static aux channel)
      - ``rhs_scratch``  = (#scratch buffers after reuse) * n_cons * C * B   (the step-body scratch)
      - ``state_scratch``= 1 state buffer * n_cons * C * B (the committed-state staging copy)
      - ``scalar_field`` = (#field solves) * C * B  (the elliptic unknown buffer)
      - ``krylov``       = (#linear solves) * 4 * C * B (Krylov needs ~4 work vectors per solve)
      - ``multigrid``    = (#field solves) * (4/3) * C * B (the geometric V-cycle hierarchy ~ 4/3 C)
      - ``halo``         = ghost_depth * perimeter * n_cons * B (the ghost ring, 2D)
      - ``mpi_buffer``   = same as halo, only when ``platform`` requests MPI (else 0)
      - ``amr_patch``    = for an ``AMR`` layout: a CONSERVATIVE per-level patch budget

    @p platform optional hint (e.g. ``"mpi"`` / ``"cpu"``) to include the MPI halo exchange buffer;
    @p layout a typed provider exposing ``capabilities().to_dict()``.  The estimator accepts the
    public ``layout='uniform'`` / ``layout='amr'`` protocol, including ``pops.layouts.AMR``; AMR
    requires explicit dimension, max-level and transition-ratio evidence.  Its figure is
    CONSERVATIVE (full refinement of every level); a tight figure needs a bind.
    """
    context = _native_memory_context()
    program = getattr(compiled, "program", None)
    cells, shape = _mesh_shape(mesh, context)
    _cons, n_cons, _params, _aux_names, n_aux, _space = _model_metadata(compiled)
    if n_cons < 0 or n_aux < 0:
        raise MemoryEstimateCapabilityError(
            "estimate_memory requires non-negative compiled component counts (got n_cons=%r, n_aux=%r)"
            % (n_cons, n_aux), field="compiled.components", actual=(n_cons, n_aux))

    # On the AMR route ``compiled`` is a CompiledModel carrying the AMR layout in its immutable
    # InstallPlan and no Program, so a bare ``estimate_memory(mesh)`` defaults to that plan. An
    # explicit layout wins.  A low-level handle without a normalized layout cannot be assigned a
    # made-up single level: callers must pass a typed Uniform or structured AMR descriptor.
    if layout is None:
        plan = getattr(compiled, "install_plan", None)
        layout = getattr(plan, "layout", None)
    if layout is None:
        raise MemoryEstimateCapabilityError(
            "estimate_memory requires an explicit typed Uniform/AMR layout or an artifact InstallPlan",
            field="layout")

    est = program.estimate() if (program is not None and hasattr(program, "estimate")) else {
        "buffer_count": 0, "heavy_kernels": 0}
    scratch_buffers = int(est.get("buffer_count", 0))
    n_field_solves = int(est.get("heavy_kernels", 0))
    n_linear_solves = sum(1 for v in getattr(program, "_values", [])
                          if v.op == "solve_linear") if program is not None else 0
    n_elliptic = max(n_field_solves - n_linear_solves, 0)

    cell_field = cells * context.real_bytes        # one scalar field with native Real precision
    state_field = n_cons * cell_field             # one full conservative-state buffer

    categories = {
        "state": state_field,
        "state_scratch": state_field,             # the committed-state staging copy
        "rhs_scratch": scratch_buffers * state_field,
        "field_output": n_elliptic * cell_field,
        "aux": n_aux * cell_field,
        "scalar_field": n_elliptic * cell_field,
        "krylov": n_linear_solves * 4 * cell_field,
        "multigrid": int(n_elliptic * (4.0 / 3.0) * cell_field),
    }

    ghost = _ghost_depth(compiled)
    nx, ny = shape
    perimeter = 2 * (nx + ny)                      # cells on the domain boundary ring (2D)
    halo = ghost * perimeter * n_cons * context.real_bytes
    categories["halo"] = halo

    requires_mpi = bool(platform) and "mpi" in str(platform).lower()
    categories["mpi_buffer"] = halo if requires_mpi else 0

    assumptions = [
        "native precision: %d bytes per cell value" % context.real_bytes,
        "native dimension=%d: %d cells = %d x %d" % (context.dimension, cells, nx, ny),
        "scratch counted AFTER the Program's static buffer-reuse report (%d buffers); the codegen "
        "may keep more, so this is a lower bound on scratch reuse" % scratch_buffers,
        "ghost halo depth assumed %d (conservative MUSCL stencil; not recorded in today's metadata)"
        % ghost,
        "Krylov work vectors assumed 4 per linear solve; multigrid hierarchy ~ 4/3 of the fine grid",
        "in-solver V-cycle / smoother traffic is NOT counted (solver-dependent, out of a static "
        "structural estimate)",
    ]

    layout_kind = "system"
    if layout is not None:
        layout_kind, amr_bytes, amr_notes = _amr_patch_budget(
            layout, state_field, cell_field, n_elliptic, context)
        if amr_bytes is not None:
            categories["amr_patch"] = amr_bytes
            assumptions.extend(amr_notes)

    if requires_mpi:
        assumptions.append("MPI halo-exchange buffer included (platform=%r); a rank-local subdomain "
                           "would be smaller -- this is the single-rank whole-domain ring" % platform)

    return MemoryEstimate(categories=categories, cells=cells, mesh_shape=shape, n_cons=n_cons,
                          n_aux=n_aux, scratch_buffers=scratch_buffers, assumptions=assumptions,
                          conservative=True, layout=layout_kind)


def _amr_patch_budget(layout: Any, state_field: Any, cell_field: Any, n_elliptic: Any,
                      context: _MemoryRuntimeContext) -> tuple:
    """A CONSERVATIVE AMR patch budget from the public layout-capability protocol (no bind).

    Returns ``(layout_kind, amr_patch_bytes, notes)``. For a ``Uniform`` layout there is no extra
    patch budget (``amr_patch_bytes`` is ``None``). For an AMR layout with ``max_levels=L`` and
    transition ratio ``r`` the
    worst case fully refines every level: a level ``k`` covering the whole domain at refinement
    ``r^k`` has ``r^(2k)`` times the base cells (2D). Summing the geometric series over the refined
    levels (1..L-1) gives the extra fine-grid footprint on top of the base level. This is an UPPER
    bound (real regrids refine a fraction of the domain); a tight figure needs a bind."""
    layout_context = _layout_context(layout, context)
    if layout_context.kind == "uniform":
        return "uniform", None, []
    max_levels = layout_context.max_levels
    if max_levels <= 1:
        return "amr", 0, ["AMR layout with a single level: no extra patch budget"]
    ratio = layout_context.ratio
    assert ratio is not None  # established by _layout_context for a refining AMR hierarchy
    # Sum r^(2k) for k = 1 .. max_levels-1 (each refined level fully covering the domain).
    refine_factor = sum(ratio ** (2 * k) for k in range(1, max_levels))
    # Each refined cell carries the same per-cell footprint as the base (state + one elliptic field).
    per_cell_levels = state_field + n_elliptic * cell_field
    amr_bytes = refine_factor * per_cell_levels
    notes = [
        "AMR estimate is CONSERVATIVE: assumes EVERY level (1..%d) fully refines the whole domain "
        "at ratio %d (worst case); a real regrid tags a fraction of cells, so the true footprint is "
        "smaller. A tight AMR figure needs a bind (the regrid pattern is data-dependent)."
        % (max_levels - 1, ratio),
        "AMR refine factor (sum of r^(2k), k=1..%d) = %d base-grid equivalents"
        % (max_levels - 1, refine_factor),
    ]
    return "amr", amr_bytes, notes

__all__ = [
    "Arguments", "MemoryEstimate", "build_arguments", "build_component_arguments",
    "build_layout_arguments",
    "build_memory_estimate"]
