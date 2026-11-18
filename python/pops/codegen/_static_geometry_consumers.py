"""The consumers that read the static geometry arrays of a mapped layout (N17, C1, first lot).

A layout whose coordinate map reads no clock is served by static arrays: measures, area vectors,
normals, centroids, distances and widths of the image of its logical box. Only the consumers below
read them; every other consumer of a block on such a layout still reads the uniform spacings of
the logical box, so it is refused here, at resolution, before any C++ (fail-closed). The list is
an allowlist: a consumer joins it when it reads the arrays and a test shows it.

* the explicit finite-volume rate of a block whose traces are the cell values (formal order one)
  and whose face law reads only what a physical normal provides: the normal flux F.n and a bound
  of the normal spectral radius, with the provider pack of each trace (the requirements that the
  face descriptor declares). The rule reads capabilities, never the name of a method (CF2), and
  serves a face of the catalogue and a face authored in source alike (``riemann.User``, the words
  of ``pops.lib.numerics`` since N3 lot 1), whose input ``n`` is then the physical unit normal of
  the face; a trace authored in source reads a stencil of the logical box and stays refused;
* transport boundaries that extend the interior state alone: a ghost state of the state, without
  a formula on its components, a characteristic closure, a field, the clock or a parameter;
* the two-point diffusion of a Program, of one component with a diagonal coefficient, on node
  positions (a graded box, of rank one to three), with conormal, value or periodic ends whose data
  reads no coordinate;
* the divergence of a stencil flux (``StencilFlux``, the realization of ``two_point()`` since N3
  lot 1) on node positions, whose row of axis a reads the two cells of its face and the spacing of
  a alone: the arrays give that spacing as the distance across the face, whose faces are
  orthogonal to the line of the centroids, and the measures of its faces and cells;
* states and linear combinations of a Program, a fixed step, initial values bound as arrays;
* the reductions that a block owns, which are raw algebra over the cell values and weigh nothing
  by a measure, and the finite assessment of an elaborated Advance that reads them (isfinite and
  the acceptance guard). A reduction that no block owns observes a field or an interaction, a
  service of the System that may weigh by the measures of the logical box, and stays refused.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from pops.time._program.stencil_flux import OPERATION as _STENCIL_OPERATION

# The operation of a stencil flux is named once, by the module that builds it.
STATIC_GEOMETRY_PROGRAM_OPS = frozenset({"state", "linear_combine", "rhs", "diffusive_rhs",
                                         _STENCIL_OPERATION, "reduce", "scalar_isfinite",
                                         "acceptance_guard"})
# What a physical face normal provides to a face law: the normal flux F.n, a bound of the normal
# spectral radius, and the provider pack of each trace.
NORMAL_FACE_CAPABILITIES = frozenset({"physical_flux", "provider_pack", "stability_bound"})
# The formal order of the traces that a mapped layout reads: the cell values.
STATIC_GEOMETRY_TRACE_ORDER = 1


class StaticGeometryConsumerError(NotImplementedError):
    """A consumer of a mapped layout that still reads the uniform spacings of its logical box."""

    def __init__(self, code: str, message: str, *, layout_id: str, evidence: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.layout_id = layout_id
        self.evidence = evidence


def _walk(value: Any, seen: set[int] | None = None):
    seen = set() if seen is None else seen
    if id(value) in seen or isinstance(value, (str, bytes, int, float, bool, type(None))):
        return
    seen.add(id(value))
    yield value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _walk(getattr(value, field.name, None), seen)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk(item, seen)
    elif isinstance(value, (tuple, list, frozenset, set)):
        for item in value:
            yield from _walk(item, seen)


def static_geometry_layouts(layout_plan: Any) -> dict[Any, Any]:
    """The layouts of a plan served by static geometry arrays, with their static reading."""
    from pops.mesh._static_geometry import layout_static_geometry

    rows = {}
    for layout in layout_plan.layouts:
        reading = layout_static_geometry(layout)
        if reading is not None:
            rows[layout.handle] = reading
    return rows


def _refuse(layout: Any, code: str, message: str, **evidence: Any) -> None:
    raise StaticGeometryConsumerError(
        code, "%s (layout %s reads static geometry arrays; this consumer reads the uniform "
        "spacings of its logical box)" % (message, layout.handle.qualified_id),
        layout_id=layout.handle.qualified_id, evidence=evidence or None)


def _check_finite_volume(layout: Any, block: Any, method: Any) -> None:
    from pops.numerics.spatial import FiniteVolume

    if type(method) is not FiniteVolume:
        _refuse(layout, "static_geometry_spatial_unavailable",
                "block %r uses %s; the static arrays are read by FiniteVolume and Diffusion only"
                % (block.name, type(method).__name__), method=type(method).__name__)
    order = (getattr(method.reconstruction, "options", None) or {}).get("formal_order")
    if order != STATIC_GEOMETRY_TRACE_ORDER:
        _refuse(layout, "static_geometry_reconstruction_unavailable",
                "block %r reconstructs traces of formal order %r; on a mapped layout the faces "
                "read the cell values (formal order %d)"
                % (block.name, order, STATIC_GEOMETRY_TRACE_ORDER), formal_order=order)
    # A face authored in source is a law of the face inputs (the traces, F.n, the normal speed
    # bound and n), served by the capability rule below; a trace authored in source is a stencil
    # of the logical box.
    if "source_identity" in (getattr(method.reconstruction, "options", None) or {}):
        _refuse(layout, "static_geometry_source_numerics_unavailable",
                "block %r authors its reconstruction in source; on a mapped layout the traces are "
                "the cell values" % block.name, part="reconstruction")
    requirements = getattr(method.riemann, "requirements", None) or {}
    required = frozenset(requirements.get("capabilities", ()))
    unserved = sorted(required - NORMAL_FACE_CAPABILITIES)
    if not required or unserved:
        _refuse(layout, "static_geometry_riemann_unavailable",
                "block %r solves faces with a law that reads %s; on a mapped layout a face law "
                "reads the normal flux F.n and a bound of its normal spectral radius only"
                % (block.name, unserved or "no declared capability"), capabilities=unserved)
    if method.sampling:
        _refuse(layout, "static_geometry_sampling_unavailable",
                "block %r samples its faces; a sampled face reads logical positions" % block.name)


def _check_stencil(layout: Any, value: Any, rank: int, kind: str) -> None:
    """A stencil flux reads, on the faces of axis a, the two cells of the face and the spacing of
    a, which the arrays of node positions give as the distance across the face."""
    from pops.model.stencil_expression import row_leaf_names
    from pops.time._program.stencil_flux import validate_stencil_flux

    block = value.block
    name = getattr(block, "name", block)
    if kind != "nodes":
        _refuse(layout, "static_geometry_stencil_unavailable",
                "block %r divides a stencil flux by the spacings of a %s of rank %d; the arrays "
                "serve the stencil of node positions, whose faces are orthogonal to the line of "
                "the centroids" % (name, kind, rank), rank=rank, kind=kind)
    declaration = validate_stencil_flux(value)
    for axis, row in enumerate(declaration.rows):
        leaves = row_leaf_names(row)
        two_cells = {tuple(-1 if d == axis else 0 for d in range(rank)), (0,) * rank}
        reads = []
        if any(symbol.name in leaves and tuple(offset) not in two_cells
               for offset, _, symbol in declaration.samples):
            reads.append("a cell beyond the two cells of its face")
        if any(symbol.name in leaves for d, symbol in enumerate(declaration.spacing) if d != axis):
            reads.append("the spacing of another axis")
        if any(symbol.name in leaves for _, _, _, symbol in declaration.field_samples):
            reads.append("a field")
        if reads:
            _refuse(layout, "static_geometry_stencil_unavailable",
                    "block %r reads, on the faces of axis %d of a stencil flux, %s; on node "
                    "positions a stencil reads the two cells of its face and the spacing of its "
                    "axis, which is the distance across the face" % (name, axis, " and ".join(reads)),
                    axis=axis, reads=reads)


def _extends_the_state(provider: Any) -> bool:
    """A ghost state of the interior state alone: no formula on its components, no closure."""
    from pops.mesh.boundaries import ClosureMode
    from pops.mesh.boundaries.ports import GhostState
    from pops.mesh.boundaries.providers import BoundaryProviderKind

    dependencies = provider.dependencies
    return (provider.kind is not BoundaryProviderKind.GHOST_FORMULA
            and bool(provider.outputs)
            and all(isinstance(row, GhostState) for row in provider.outputs)
            and dependencies.characteristic.mode is ClosureMode.NONE
            and bool(dependencies.states) and not dependencies.fields
            and not dependencies.time and not dependencies.runtime_params)


def _diagonal(law: Any) -> bool:
    """Every off-diagonal entry of the spatial tensor of the law is an exact zero."""
    from pops.model.expression_language import Const

    return all(axis == column or (isinstance(value, Const) and value.value == 0)
               for tensor in law.component_coefficients
               for axis, row in enumerate(tensor) for column, value in enumerate(row))


def _check_diffusion(layout: Any, block: Any, method: Any, rank: int, kind: str) -> None:
    from pops.model.spaces import StateSpace

    law = method.law
    if any(type(space) is not StateSpace for space in law.inputs):
        _refuse(layout, "static_geometry_diffusion_unavailable",
                "block %r diffuses with a law that reads a field or an auxiliary input; the "
                "static arrays serve laws of the state alone" % block.name)
    if method.transport is not None:
        _refuse(layout, "static_geometry_diffusion_unavailable",
                "block %r couples diffusion with transport faces in one balance" % block.name)
    if kind != "nodes":
        _refuse(layout, "static_geometry_diffusion_unavailable",
                "block %r diffuses on a %s of rank %d; the diffusion of the static arrays reads "
                "node positions (a graded box), whose faces are orthogonal to the line of the "
                "centroids" % (block.name, kind, rank), rank=rank, kind=kind)
    # Capabilities, never a method name (CF2): a face that reads its two cells alone (ghost
    # depth one), one component, a diagonal spatial tensor.
    if getattr(method, "ghost_depth", None) != 1 or len(law.variables) != 1 or not _diagonal(law):
        _refuse(layout, "static_geometry_diffusion_unavailable",
                "block %r diffuses a vector, with a full tensor or through a wider face; the "
                "static arrays serve the two-point diffusion of one component with a diagonal "
                "coefficient, whose faces are orthogonal to the line of the centroids on node "
                "positions" % block.name)
    for boundary in law.boundaries:
        if any(float(value) != 0.0 for value in (boundary.slope or ())):
            _refuse(layout, "static_geometry_boundary_data_unavailable",
                    "block %r has boundary data that reads a coordinate; on a mapped layout it "
                    "would read a logical position" % block.name)


def validate_static_geometry_consumers(plan: Any) -> None:
    """Refuse, before any C++, every consumer of a mapped layout that the arrays do not serve."""
    from pops.codegen.program_lowerability import all_ops
    from pops.fields.aux import AnalyticAux
    from pops.mesh.boundaries.providers import BoundaryProviderKind
    from pops.numerics.diffusion import Diffusion
    from pops.time._step.strategy import FixedDt

    layouts = static_geometry_layouts(plan.layout_plan)
    if not layouts:
        return
    by_handle = {row.handle: row for row in plan.layout_plan.layouts}
    if len(plan.layout_plan.layouts) != 1:
        _refuse(next(iter(by_handle.values())), "static_geometry_multi_layout_unavailable",
                "a mapped layout is served alone in one runtime")
    handle, reading = next(iter(layouts.items()))
    layout = by_handle[handle]
    ops = sorted({value.op for value in all_ops(plan.time)} - STATIC_GEOMETRY_PROGRAM_OPS)
    if ops:
        _refuse(layout, "static_geometry_consumer_unavailable",
                "Program operation(s) %s" % ops, operations=ops)
    for value in all_ops(plan.time):
        if value.op == "rhs" and (value.attrs.get("named_fluxes") or any(
                source != "default" for source in (value.attrs.get("sources") or ()))):
            _refuse(layout, "static_geometry_consumer_unavailable",
                    "a rate with named fluxes or named sources")
        if value.op == _STENCIL_OPERATION:
            _check_stencil(layout, value, reading.rank, reading.kind)
        if value.op == "reduce" and value.block is None:
            _refuse(layout, "static_geometry_consumer_unavailable",
                    "a reduction %r that no block owns, which observes a field or an interaction "
                    "through the System" % (value.attrs.get("kind"),),
                    operations=["reduce"], kind=value.attrs.get("kind"))
    strategy = getattr(plan.time, "_step_strategy", None)
    if type(strategy) is not FixedDt:
        _refuse(layout, "static_geometry_step_unavailable",
                "step strategy %s; a stability step reads the spacings of the logical box, so a "
                "mapped layout takes FixedDt" % type(strategy).__name__)
    if plan.field_plans or plan.program_field_plans or plan.component_inputs \
            or plan.consumer_graph is not None:
        _refuse(layout, "static_geometry_consumer_unavailable",
                "fields, auxiliary inputs or output consumers")
    initial = plan.initial_condition_plan
    for binding in () if initial is None else initial.bindings:
        route = binding.source.options.to_data().get("native_route")
        if route != "bound_level_zero":
            _refuse(layout, "static_geometry_initial_unavailable",
                    "initial condition route %r; on a mapped layout initial values are bound as "
                    "arrays (BindArray) of cell averages" % (route,), route=route)
    for block in plan.blocks:
        module = getattr(block.model, "module", None)
        producers = {} if module is None else module.aux_providers()
        if any(isinstance(producer, AnalyticAux) for producer in producers.values()):
            _refuse(layout, "static_geometry_consumer_unavailable",
                    "block %r has an analytic auxiliary input, which reads the cell coordinates "
                    "of the logical box" % block.name)
        numerics = block.numerics
        if numerics.sources or numerics.fields or numerics.interfaces \
                or numerics.principal_groups:
            _refuse(layout, "static_geometry_consumer_unavailable",
                    "block %r has sources, fields, interfaces or principal groups" % block.name)
        for rate in numerics.rates:
            if isinstance(rate.method, Diffusion):
                _check_diffusion(layout, block, rate.method, reading.rank, reading.kind)
            else:
                _check_finite_volume(layout, block, rate.method)
        unserved = sorted({node.kind.value for node in _walk(numerics.boundaries)
                           if isinstance(getattr(node, "kind", None), BoundaryProviderKind)
                           and not _extends_the_state(node)})
        if unserved:
            _refuse(layout, "static_geometry_boundary_unavailable",
                    "block %r has transport boundaries %s; on a mapped layout a boundary is served "
                    "when it extends the interior state alone, since a formula on its components, "
                    "a wall or a characteristic closure reads the logical axis, not the face normal"
                    % (block.name, unserved), boundaries=unserved)


__all__ = ["NORMAL_FACE_CAPABILITIES", "STATIC_GEOMETRY_PROGRAM_OPS", "StaticGeometryConsumerError",
           "static_geometry_layouts", "validate_static_geometry_consumers"]
