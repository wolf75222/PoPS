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
* transport boundaries that extend the interior state alone, including a generated ghost that
  copies each component directly from the nearest interior cell at every depth;
* a named cell source whose expression reads only its conservative state and numbers, both in a
  joint rate and on its own. A StateStorage rate method may select these sources alone;
* the two-point diffusion of a Program, of one component with a diagonal coefficient, on node
  positions (a graded box, of rank one to three) and, with one coefficient on every logical axis,
  on a chart of rank two (DN35), with conormal, value or periodic ends whose data reads no
  coordinate;
* the divergence of a stencil flux (``StencilFlux``, the realization of ``two_point()`` since N3
  lot 1), whose row of axis a reads the two cells of its face and the spacing of a alone, and on a
  chart is one law on every logical axis (DN35): the arrays give that spacing as the distance
  across the face, and the measures of its faces and cells;
* the centred differences of a Program (N17 L2b-2, DN38): the slope g_K of a cell solves
  delta X g = delta e on the differences of the centroids of its neighbours, the stored centroids
  inside, translated by the seam across a periodic axis and reflected through the centroid of the
  end face, 2 m_f - c_K, at a physical end (``pops::centroid_slope``, the header function that the
  generated cell read of N15 will share); its values are those of the boundary session of the
  Program. With it, the scratch that a Program allocates for it and the cellwise algebra of
  Program values (a pointwise expression, of which the cell mean of a computed cell is one), which
  reads values, scalars and literals and never a coordinate; the Laplacian and the divergence of
  a Program read the uniform spacings and stay refused;
* states and linear combinations of a Program, a fixed step, initial values bound as arrays;
* the reductions that a block owns, which are raw algebra over the cell values and weigh nothing
  by a measure, and the finite assessment of an elaborated Advance that reads them (isfinite and
  the acceptance guard). A reduction that no block owns observes a field or an interaction, a
  service of the System that may weigh by the measures of the logical box, and stays refused.

A two-point face reads the difference of its two cells over the distance of their centroids along
the normal of the face. The faces of node positions are orthogonal to the line of the centroids; a
face of a chart takes that line for its normal (K-orthogonality), which is measured, not required:
the arrays publish the largest sine between the two beside the closure ratio, and ``pops.inspect``
states the assumption (DN36). A law that reads the normal through n.n = 1 alone (a scalar
coefficient, the same row on every logical axis) gives, with the axis normal, the flux density
along the physical normal; one that reads more of it (diag(k_x, k_y), whose normal flux on a chart
is (k_x n_x^2 + k_y n_y^2) du/dn) is refused on a chart (DN35).
"""
from __future__ import annotations

import dataclasses
from typing import Any

from pops.time._program.local import CENTRED_DIFFERENCES as _CENTRED_DIFFERENCES
from pops.time._program.stencil_flux import OPERATION as _STENCIL_OPERATION

# The operations of a stencil flux and of centred differences are named once, by the modules that
# build them.
STATIC_GEOMETRY_PROGRAM_OPS = frozenset({"state", "linear_combine", "rhs", "source", "diffusive_rhs",
                                         _STENCIL_OPERATION, "reduce", "scalar_isfinite",
                                         "acceptance_guard", _CENTRED_DIFFERENCES, "scalar_field",
                                         "pointwise_expression"})
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


def _renamed(key: Any, names: dict) -> Any:
    """The structural key ``key`` of an expression, its variables renamed by ``names``."""
    if type(key) is tuple:
        if len(key) == 3 and key[0] == "var" and key[2] in names:
            return ("var", key[1], names[key[2]])
        return tuple(_renamed(part, names) for part in key)
    return key


def _one_law(declaration: Any, rank: int) -> bool:
    """The rows of a stencil flux are one law on every logical axis (DN35): the row of axis a,
    the samples of its two cells and its spacing renamed, is the row of every other axis."""
    from pops._ir.visitors import _key

    rows = []
    for axis, row in enumerate(declaration.rows):
        left = tuple(-1 if d == axis else 0 for d in range(rank))
        names = {declaration.spacing[axis].name: "spacing"}
        for offset, component, symbol in declaration.samples:
            if tuple(offset) == left:
                names[symbol.name] = "left %d" % component
            elif tuple(offset) == (0,) * rank:
                names[symbol.name] = "right %d" % component
        rows.append(tuple(_renamed(_key(value), names) for value in row))
    return all(row == rows[0] for row in rows[1:])


def _check_stencil(layout: Any, value: Any, rank: int, kind: str) -> None:
    """A stencil flux reads, on the faces of axis a, the two cells of the face and the spacing of
    a, which the arrays give as the distance across the face. On a chart its rows are one law on
    every logical axis, which reads the normal of a face through n.n = 1 alone, so that the row of
    the axis normal is the flux density along the physical normal (DN35)."""
    from pops.model.stencil_expression import row_leaf_names
    from pops.time._program.stencil_flux import validate_stencil_flux

    block = value.block
    name = getattr(block, "name", block)
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
                    "block %r reads, on the faces of axis %d of a stencil flux, %s; on static "
                    "geometry arrays a stencil reads the two cells of its face and the spacing of "
                    "its axis, which is the distance across the face"
                    % (name, axis, " and ".join(reads)), axis=axis, reads=reads)
    if kind == "chart" and not _one_law(declaration, rank):
        _refuse(layout, "static_geometry_stencil_unavailable",
                "block %r writes its stencil flux as a different law on each logical axis of a "
                "chart of rank %d; the normal of a face of a chart is not its axis, and a stencil "
                "is served there as one law on every logical axis, which reads the normal through "
                "n.n = 1 (a scalar coefficient)" % (name, rank), rank=rank, kind=kind)


def _proved_ghost_copies(boundaries: Any) -> set[int]:
    """Find copies whose actual transport condition owns the physical ghost production."""
    from pops.boundary.transport import ResolvedTransportBoundarySet, SourceGhost
    from pops.mesh.boundaries import GhostProducerPlan
    from pops.mesh.boundaries.ports import GhostState
    from pops.mesh.boundaries.providers import BoundaryProviderKind

    admitted = set()
    refused = set()
    for plan in boundaries:
        if type(plan) is not GhostProducerPlan:
            continue
        authority = plan.execution_authority
        for production in plan.productions:
            region = production.region
            for provider in production.producer.boundary_providers:
                if provider.kind is not BoundaryProviderKind.GHOST_FORMULA:
                    continue
                matches = (
                    [condition for condition in authority.conditions if condition.provider is provider]
                    if type(authority) is ResolvedTransportBoundarySet else []
                )
                if len(matches) != 1 or region.boundary is None:
                    refused.add(id(provider))
                    continue
                condition = matches[0]
                source = condition.source_ghost
                geometry = condition.geometry
                orientation = region.boundary.orientation
                deps = provider.dependencies
                if (
                    type(source) is SourceGhost
                    and source.state == condition.state == region.subject
                    and deps.states == (region.subject,)
                    and geometry.domain_geometry_id == authority.domain_geometry_id
                    and (geometry.axis.index, geometry.side.value)
                    == (orientation.axis, orientation.side.value)
                    and condition.requirement.ghost_depth <= source.depth
                    and orientation.axis < len(region.depth.depth)
                    and region.depth.depth[orientation.axis] <= source.depth
                    and deps.representation.source == deps.representation.target
                    and deps.representation.converter is None
                    and all(type(output) is GhostState and output.subject == region.subject
                            and output.boundary == region.boundary for output in provider.outputs)
                    and any(binding.provider is provider and binding.need.subject == region.subject
                            and binding.need.boundary == region.boundary
                            for binding in authority.plan.bindings)
                    and not source.runtime_captures
                    and source._copies_nearest_interior()
                ):
                    admitted.add(id(provider))
                else:
                    refused.add(id(provider))
    return admitted - refused


def _extends_the_state(provider: Any, ghost_copies: set[int]) -> bool:
    """A ghost state of the interior state alone: no formula on its components, no closure."""
    from pops.mesh.boundaries import ClosureMode
    from pops.mesh.boundaries.ports import GhostState
    from pops.mesh.boundaries.providers import BoundaryProviderKind

    dependencies = provider.dependencies
    return ((provider.kind is not BoundaryProviderKind.GHOST_FORMULA or id(provider) in ghost_copies)
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


def _isotropic(law: Any) -> bool:
    """Every diagonal entry of the spatial tensor of the law is one expression (DN35)."""
    from pops._ir.visitors import _key

    for tensor in law.component_coefficients:
        entries = [_key(row[axis]) for axis, row in enumerate(tensor)]
        if any(entry != entries[0] for entry in entries[1:]):
            return False
    return True


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
    # Capabilities, never a method name (CF2): a face that reads its two cells alone (ghost
    # depth one), one component, a diagonal spatial tensor.
    if getattr(method, "ghost_depth", None) != 1 or len(law.variables) != 1 or not _diagonal(law):
        _refuse(layout, "static_geometry_diffusion_unavailable",
                "block %r diffuses a vector, with a full tensor or through a wider face; the "
                "static arrays serve the two-point diffusion of one component with a diagonal "
                "coefficient" % block.name)
    if kind == "chart" and not _isotropic(law):
        _refuse(layout, "static_geometry_diffusion_unavailable",
                "block %r diffuses on a chart of rank %d with a coefficient that differs by "
                "logical axis; the normal of a face of a chart is not its axis, and its two-point "
                "face is served with one coefficient on every logical axis, whose normal flux "
                "k du/dn reads the normal through n.n = 1" % (block.name, rank), rank=rank,
                kind=kind)
    for boundary in law.boundaries:
        if any(float(value) != 0.0 for value in (boundary.slope or ())):
            _refuse(layout, "static_geometry_boundary_data_unavailable",
                    "block %r has boundary data that reads a coordinate; on a mapped layout it "
                    "would read a logical position" % block.name)


def _source_reads_state(module: Any, state: Any, name: str) -> bool:
    """Prove the inputs of an exact named source in its owner's operator registry."""
    from pops._ir.control_expr import Rounded, Where
    from pops._ir.expr import Abs, BooleanNot, Const, Exp, Neg, Sign, Sqrt, UnaryMath, Var, _Bin
    from pops._ir.primitive_expansion import expand_primitive_recipes
    from pops._ir.quantity import QuantityRef
    from pops._ir.visitors import _children

    if state is None or name == "default":
        return False
    spaces = module.state_spaces()
    space = spaces.get(state.local_id)
    if space is None or state != module.state_handle(space):
        return False
    try:
        operator = module.operator_registry().get(name)
    except KeyError:
        return False
    if (operator.kind != "local_source" or operator.signature.inputs != (space,)
            or operator.requirements or not isinstance(operator.body, tuple)
            or len(operator.body) != len(space.components)):
        return False
    # Follow only the owned primitive recipes reachable from this body. A provider or
    # parameter inside a recipe remains a typed leaf after expansion.
    pending = list(expand_primitive_recipes(operator.body, module.primitive_recipes()))
    seen = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if type(node) is Const:
            continue
        if type(node) is QuantityRef and node.handle == state:
            continue
        if (type(node) is Var and node.kind == "cons" and len(spaces) == 1
                and node.name in space.components):
            continue
        if not isinstance(node, (_Bin, Abs, BooleanNot, Exp, Neg, Rounded, Sign, Sqrt,
                                 UnaryMath, Where)):
            return False
        children = _children(node)
        if not children:
            return False
        pending.extend(children)
    return True


def _source_module(block: Any) -> Any:
    from pops.model.module import Module

    model = block.model
    return model if isinstance(model, Module) else getattr(model, "module", None)


def _check_program_sources(layout: Any, blocks: dict[str, Any], value: Any, names: Any) -> None:
    from pops.model.handles import OperatorHandle
    from pops.problem.handles import BlockHandle

    owner = value.block
    block = (blocks.get(str(owner.instance_owner_path.canonical()))
             if type(owner) is BlockHandle else None)
    module = None if block is None else _source_module(block)
    operator = value.attrs.get("operator_handle")
    if (block is not None and module is not None and len(value.inputs) == 1
            and value.inputs[0].block == owner
            and (operator is None or isinstance(operator, OperatorHandle)
                 and operator.owner_path.canonical().to_data() in
                 (block.model_owner, block.instance_owner))):
        state = getattr(value.inputs[0].state_ref, "declaration_ref", None)
        if all(_source_reads_state(module, state, name) for name in names):
            return
    _refuse(layout, "static_geometry_consumer_unavailable",
            "Program %s sources %r require a named law of their own conservative state and "
            "numbers on static geometry" % (value.op, tuple(names)), operation=value.op)


def _check_source_selection(layout: Any, block: Any, rate: Any) -> None:
    from pops.model.expression_language import source_balance_supported
    from pops.model.handles import OperatorHandle
    from pops.physics.board_handles import SourceHandle

    module = _source_module(block)
    if module is not None and rate.rate.owner_path.canonical().to_data() == block.instance_owner:
        registered = module.operator_handle(rate.rate.registered_operator_name)
        contract = module.rate_contract(registered)
        state, sources = contract["state"], contract["sources"]
        balance_contract = getattr(block.model, "balance_contract", None)
        if callable(balance_contract):
            try:
                view: Any = balance_contract(registered)
            except ValueError:
                view = None
            if (not source_balance_supported(view) or not view.occurrences
                    or any(term.target != state or term.payload not in sources
                           for term in view.occurrences)):
                sources = ()
        if not contract["flux"] and not contract.get("nonconservative_products") and sources:
            for source in sources:
                if not isinstance(source, (SourceHandle, OperatorHandle)):
                    break
                if isinstance(source, OperatorHandle) and source.kind != "local_source":
                    break
                name = (source.reg_name if isinstance(source, SourceHandle)
                        else source.registered_operator_name)
                if source.owner_path != module.owner_path or not _source_reads_state(module, state, name):
                    break
            else:
                return
    _refuse(layout, "static_geometry_spatial_unavailable",
            "block %r uses StateStorage for a selection other than named sources of its own "
            "conservative state and numbers" % block.name, method="StateStorage")


def validate_static_geometry_consumers(plan: Any) -> None:
    """Refuse, before any C++, every consumer of a mapped layout that the arrays do not serve."""
    from pops.codegen.program_lowerability import all_ops
    from pops.codegen.program_emit_kernels import _named_fluxes
    from pops.fields.aux import AnalyticAux
    from pops.mesh.boundaries.providers import BoundaryProviderKind
    from pops.numerics.diffusion import Diffusion
    from pops.numerics.state_storage import StateStorage
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
    blocks = {block.instance_owner_qid: block for block in plan.blocks}
    ops = sorted({value.op for value in all_ops(plan.time)} - STATIC_GEOMETRY_PROGRAM_OPS)
    if ops:
        _refuse(layout, "static_geometry_consumer_unavailable",
                "Program operation(s) %s" % ops, operations=ops)
    for value in all_ops(plan.time):
        if value.op == "rhs":
            if value.attrs.get("named_fluxes") or _named_fluxes(value):
                _refuse(layout, "static_geometry_consumer_unavailable", "a rate with named fluxes")
            sources = value.attrs.get("sources") or ()
            if any(source != "default" for source in sources):
                _check_program_sources(layout, blocks, value, sources)
        if value.op == "source":
            _check_program_sources(layout, blocks, value, (value.attrs["source"],))
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
            elif type(rate.method) is StateStorage:
                _check_source_selection(layout, block, rate)
            else:
                _check_finite_volume(layout, block, rate.method)
        ghost_copies = _proved_ghost_copies(numerics.boundaries)
        unserved = sorted({node.kind.value for node in _walk(numerics.boundaries)
                           if isinstance(getattr(node, "kind", None), BoundaryProviderKind)
                           and not _extends_the_state(node, ghost_copies)})
        if unserved:
            _refuse(layout, "static_geometry_boundary_unavailable",
                    "block %r has transport boundaries %s; on a mapped layout a boundary is served "
                    "when it extends the interior state alone, since a formula on its components, "
                    "a wall or a characteristic closure reads the logical axis, not the face normal"
                    % (block.name, unserved), boundaries=unserved)


__all__ = ["NORMAL_FACE_CAPABILITIES", "STATIC_GEOMETRY_PROGRAM_OPS", "StaticGeometryConsumerError",
           "static_geometry_layouts", "validate_static_geometry_consumers"]
