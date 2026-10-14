"""One explicit nonlocal spatial map; publication belongs to the surrounding Program."""
from typing import TYPE_CHECKING
from pops.time.values import ProgramValue, _resolve_handle
from pops.time._program.value_validation import require_top_level, validate_input_regions
from pops.time._authoring import atomic_authoring
from pops.model.spaces import FieldSpace
from pops._ir.quantity import PhysicalDimension


def _midpoint_spaces(field, state):
    """Authenticate an explicit piecewise-constant sampling map, never a cast."""
    from pops.model.spaces import StateSpace
    if field.vtype != "scalar_field" or type(field.space) is not FieldSpace:
        raise TypeError("midpoint sampling requires a typed computed cell field")
    if state.vtype != "state" or type(state.space) is not StateSpace:
        raise TypeError("midpoint sampling requires an exact State template")
    if field.prog is not state.prog or field.block != state.block or field.block is None:
        raise ValueError("midpoint sampling cannot change Program or block ownership")
    if field.point != state.point:
        raise ValueError("midpoint sampling requires the same exact evaluation point")
    if field.space.sampling != "cell_center" or state.space.sampling != "cell_average":
        raise ValueError("midpoint sampling requires cell_center and cell_average representations")
    for key in ("layout", "centering", "frame", "clock", "support"):
        if getattr(field.space, key) != getattr(state.space, key):
            raise ValueError("midpoint sampling changes physical " + key)
    from pops.representations import _data_identity
    if _data_identity(field.space.representation_data) != _data_identity(state.space.representation_data):
        raise ValueError("midpoint sampling changes physical representation_data")
    if (field.space.layout != "cell" or field.space.centering != "cell"
            or len(field.space.components) != len(state.space.components)
            or field.space.units != state.space.units
            or field.space.value_shape != state.space.value_shape):
        raise ValueError("midpoint sampling changes component shape or units")


def validate_cell_field(value):
    """Recheck loaded/generated midpoint samples against their actual SSA input."""
    from pops.time.values import _Coeff
    if (value.op != "linear_combine" or len(value.inputs) != 1
            or value.attrs.get("cell_field_projection") != "pops.cell-midpoint-sampling@1"
            or value.attrs.get("coeffs") != (_Coeff({0: 1}).to_polynomial(),)
            or value.state_ref is not None):
        raise ValueError("cell field lost its declared midpoint reconstruction")
    _midpoint_spaces(value, value.inputs[0])
    return value.inputs[0]


def validate_cell_mean(value):
    from pops.time.expressions import encode_expressions
    if (value.op != "pointwise_expression" or value.vtype != "state"
            or len(value.inputs) != 2
            or value.attrs.get("cell_mean_projection") != "pops.cell-midpoint-quadrature@1"):
        raise ValueError("cell mean lost its declared midpoint quadrature")
    field, template = value.inputs
    _midpoint_spaces(field, template)
    if (value.space != template.space or value.block != template.block
            or value.point != template.point or value.state_ref != template.state_ref):
        raise ValueError("cell mean changed its output authority")
    encoded, nodes, inputs = encode_expressions(
        tuple(field[i] for i in range(len(field.space.components))), value.prog)
    if (inputs != (field,) or value.attrs.get("expressions") != encoded
            or value.attrs.get("expression_nodes") != nodes):
        raise ValueError("cell mean changed its quadrature evaluations")
    return template


def validate_spatial_scratch(value):
    if (value.op != "scalar_field" or len(value.inputs) != 1
            or value.attrs.get("spatial_port_storage") != "pops.computed-cell-field@1"):
        raise ValueError("computed cell scratch lost its storage contract")
    source = value.inputs[0]
    if (source.vtype != "scalar_field" or type(source.space) is not FieldSpace
            or value.vtype != "scalar_field" or value.space != source.space
            or source.block is None or value.block != source.block
            or value.point != source.point or value.state_ref is not None
            or value.attrs.get("ncomp") != len(source.space.components)):
        raise ValueError("computed cell scratch changed its owner, point or representation")
    return source


def validate_computed_laplacian(value):
    """Authenticate the pure computed-field stencil, including its actual storage input."""
    if value.op != "laplacian" or len(value.inputs) != 2:
        raise ValueError("computed Laplacian lost its stencil inputs")
    output, source = value.inputs
    validate_spatial_scratch(output)
    from pops.time.field_sampling import scalar_co_location
    scalar_co_location(output, source, "computed Laplacian")
    if (output is source
            or value.vtype != "scalar_field" or value.space != source.space
            or value.block != source.block or value.point != source.point
            or value.state_ref is not None
            or value.attrs.get("ncomp") != len(source.space.components)):
        raise ValueError("computed Laplacian changed its prototype, authority or component shape")
    return output, source


def _dimension(data, *, allow_unknown=True):
    if data is None:
        if not allow_unknown:
            raise ValueError("declared spatial measure coordinate units cannot be unknown")
        return None
    from pops.time.canonical_data import _json_ready
    image = _json_ready(data)
    if type(image) is not dict or set(image) != {"kind", "powers"} or image["kind"] != "physical_dimension" or type(image["powers"]) is not list:
        raise ValueError("invalid spatial interaction physical dimension image")
    for power in image["powers"]:
        if (type(power) is not list or len(power) != 3 or type(power[0]) is not str
                or type(power[1]) is not int or type(power[2]) is not int):
            raise ValueError("spatial interaction dimension powers require exact integer rationals")
    dimension = PhysicalDimension.from_data(image)
    if dimension.to_data() != image:
        raise ValueError("spatial interaction dimension image is not canonical")
    return dimension


def _units(source, output, components, kernel_units, coordinate_units, dimension):
    if coordinate_units and len(coordinate_units) != dimension:
        raise ValueError("cell measure coordinate units differ from kernel dimension")
    for selected, actual in zip(components, output.units, strict=True):
        rho = source.space.units[selected]
        if rho is None or kernel_units is None or not coordinate_units:
            if actual is not None:
                raise ValueError("interaction output units cannot be invented from unknown physical units")
            continue
        powers = {}
        for unit in (rho, kernel_units, *coordinate_units):
            for name, power in unit.powers:
                powers[name] = powers.get(name, 0) + power
        if actual != PhysicalDimension(tuple(powers.items())):
            raise ValueError("interaction output units differ from W*rho*dmu")


def _spaces(source, output, components):
    if type(output) is not FieldSpace or len(output.components) != len(components):
        raise TypeError("spatial interaction requires an explicit physical output FieldSpace")
    if source.space.layout != "cell" or source.space.centering != "cell":
        raise ValueError("cell-midpoint interaction requires a cell-centered source")
    if output.sampling != "cell_center" or output.value_shape != (len(components),):
        raise ValueError("cell-midpoint interaction output requires a cell_center vector sampling")
    for attr in ("layout", "centering", "frame", "clock", "support"):
        if getattr(output, attr) != getattr(source.space, attr):
            raise ValueError("spatial interaction output changes " + attr)


def _source_point(source):
    from pops.time.points import TimePoint
    # A calculated candidate may carry a later point. Its State.n leaves must
    # still read n; a linear combination must not hide a relabelled carrier.
    pending = [source]
    visited = set()
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if node.op == "state" and node.point != TimePoint(node.clock, 0):
            raise ValueError("spatial interaction State.n cannot be relabelled to another point")
        canonical = node.prog._canonical_value(node)
        if canonical is not node:
            pending.append(canonical)
        pending.extend(node.inputs)
        pending.extend(node.prog._subblock_value_refs(node))



def _closed_source_contract(program, source, owner, output):
    from .global_history_storage import storage_contract
    from pops.fields._observation_contract import validate_field_observation
    width, selected, solve = validate_field_observation(source)
    if solve.op != "solve_spatial_field":
        raise NotImplementedError("closed composite source requires the original full-residual provider")
    metadata = storage_contract(program, source, owner)
    witness = next(value for value in solve.inputs if value.block == metadata["layout_witness"] and value.space is not None)
    if type(output) is not FieldSpace or len(output.components) != 1 or output.value_shape != (1,) or output.sampling != "cell_center":
        raise ValueError("closed original interaction requires one explicit cell_center output FieldSpace")
    for attr in ("layout", "centering", "frame", "clock", "support"):
        if getattr(output, attr) != getattr(witness.space, attr):
            raise ValueError("closed original interaction output changes solved " + attr)
    # Field unknown units are not encoded by the current physical FieldProblem.
    # The storage State's units are never an authority for the global unknown.
    if any(unit is not None for unit in output.units):
        raise ValueError("closed original interaction output units cannot be inferred from unknown field units")
    return {**metadata, "contract":"pops.completed-original-field-source@1",
            "tuple_width":width, "tuple_component":selected, "source_value_id":source.id, "solve_value_id":solve.id}


def validate_closed_issuances(program):
    from .global_history_storage import _equal_metadata
    records = getattr(program, "_closed_field_interaction_issuance", {})
    nodes = {value.id:value for value in program._values}
    for consumer in program._values:
        if any(item.id in records for item in consumer.inputs) and consumer.op != "reduce":
            raise NotImplementedError("completed interaction @1 exposes read-only full towers and scalar reductions; gradient/field/history consumers need their own authenticated tower/ghost port")
    for key, (_original, issued) in records.items():
        node = nodes.get(key)
        if node is None or node.op != "spatial_interaction" or node.attrs.get("contract") != "pops.spatial-interaction@3":
            raise ValueError("closed interaction lost its originally issued source snapshot contract")
        metadata = node.attrs.get("closed_field_source")
        expected = _closed_source_contract(program, node.inputs[0], issued.metadata["owner_block"], node.space)
        if not _equal_metadata(metadata, issued.metadata) or not _equal_metadata(expected, issued.metadata):
            raise ValueError("closed interaction changed its originally issued source/storage authority")
        if node.block is not None or node.state_ref is not None or node.point != node.inputs[0].point:
            raise ValueError("closed interaction cannot borrow physical State ownership")


def transfer_closed_issuances(source, target, remap):
    from types import MappingProxyType
    from .global_history_storage import _IssuedStorage, _ISSUER
    current = {}
    for _key, (node, issued) in getattr(source, "_closed_field_interaction_issuance", {}).items():
        mapped = remap({"value":node, "clock":issued.metadata["clock"],
                        "point":issued.metadata["point"], "region":issued.metadata["region"]})["value"]
        metadata = remap(issued.metadata)
        metadata["source_value_id"] = mapped.inputs[0].id
        metadata["solve_value_id"] = mapped.inputs[0].inputs[0].inputs[0].inputs[0].id
        attrs = dict(mapped.attrs)
        attrs["closed_field_source"] = metadata
        mapped = target._replace_value(mapped, attrs=attrs)
        current[mapped.id] = (mapped, _IssuedStorage(metadata, issuer=_ISSUER))
    if current:
        object.__setattr__(target, "_closed_field_interaction_issuance", MappingProxyType(current))

def interaction_contract(value):
    from pops.fields.spatial_interaction import kernel_cpp
    if value.op != "spatial_interaction" or value.vtype != "scalar_field" or len(value.inputs) != 1:
        raise ValueError("invalid spatial interaction SSA contract")
    attrs = value.attrs
    required = {"contract", "kernel", "measure", "quadrature", "realization", "max_workspace_bytes", "components", "ncomp", "source_scope"}
    closed_v3 = attrs.get("contract") == "pops.spatial-interaction@3"
    if closed_v3:
        required.add("closed_field_source")
    history_v2 = attrs.get("contract") == "pops.spatial-interaction@2"
    if history_v2:
        required.add("history_source")
    if set(attrs) != required or attrs["contract"] not in ("pops.spatial-interaction@1", "pops.spatial-interaction@2", "pops.spatial-interaction@3"):
        raise ValueError("invalid spatial interaction contract/version")
    kernel = attrs["kernel"]
    if set(kernel) != {"contract", "dimension", "tree", "units"} or kernel["contract"] != "pops.spatial-interaction-kernel@1":
        raise ValueError("invalid spatial interaction kernel contract")
    dimension = kernel["dimension"]
    if type(dimension) is not int or not 1 <= dimension <= 3:
        raise ValueError("invalid spatial interaction dimension")
    cpp = kernel_cpp(kernel["tree"], dimension)
    measure = attrs["measure"]
    if (set(measure) != {"contract", "coordinate_units"} or measure["contract"] != "pops.cell-volume-eb@1" or attrs["quadrature"] != "pops.cell-midpoint@1"
            or attrs["realization"] != "pops.direct-spatial-interaction@1"):
        raise ValueError("invalid spatial interaction measure/quadrature/realization")
    budget = attrs["max_workspace_bytes"]
    if not isinstance(budget, dict) and not hasattr(budget, "keys"):
        raise ValueError("invalid spatial interaction budget")
    if set(budget) != {"uint64_hex"} or type(budget["uint64_hex"]) is not str:
        raise ValueError("invalid spatial interaction budget")
    raw = budget["uint64_hex"]
    if len(raw) != 16 or any(c not in "0123456789abcdef" for c in raw) or int(raw, 16) == 0:
        raise ValueError("invalid spatial interaction budget")
    source = value.inputs[0]
    if closed_v3:
        validate_closed_issuances(value.prog)
        if attrs["source_scope"] != "completed_original" or tuple(attrs["components"]) != (0,) or any(type(component) is not int for component in attrs["components"]) or type(attrs["ncomp"]) is not int or attrs["ncomp"] != 1:
            raise ValueError("closed original source component/scope changed")
        _dimension(kernel["units"])
        tuple(_dimension(unit, allow_unknown=False) for unit in measure["coordinate_units"])
        if measure["coordinate_units"] and len(measure["coordinate_units"]) != dimension:
            raise ValueError("closed interaction coordinate units differ from kernel dimension")
        return source, dimension, cpp, int(raw, 16), (0,)
    if history_v2:
        from .spatial_history_source import history_source_contract, exact_image
        if source.op != "history" or exact_image(attrs["history_source"]) != exact_image(history_source_contract(source)):
            raise ValueError("nonlocal history seed closure changed its authority")
    _source_point(source)
    canonical = source.prog._canonical_value(source)
    _source_point(canonical)
    if canonical.point != source.point or canonical.space != source.space or canonical.block != source.block:
        raise ValueError("spatial interaction source authority changed after issue")
    if attrs["source_scope"] not in ("issued", "accepted") or (attrs["source_scope"] == "accepted" and source.op != "state"):
        raise ValueError("accepted composite interaction requires the issued State.n carrier")
    if source.op == "history" and "history_contract" not in source.attrs:
        raise ValueError("composite interaction history requires an authenticated sample/point contract")
    width = len(source.space.components) if source.space is not None else None
    components = tuple(attrs["components"])
    if (source.vtype != "state" or source.block != value.block or source.point != value.point
            or not components or len(set(components)) != len(components)
            or any(type(c) is not int or width is None or not 0 <= c < width for c in components)
            or type(attrs["ncomp"]) is not int or attrs["ncomp"] != len(components)):
        raise ValueError("spatial interaction source/owner/point/components changed")
    _spaces(source, value.space, components)
    _units(source, value.space, components, _dimension(kernel["units"]), tuple(_dimension(unit, allow_unknown=False) for unit in measure["coordinate_units"]), dimension)
    return source, dimension, cpp, int(raw, 16), components


if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramSpatialInteraction(_ProgramBase):
    @atomic_authoring
    def cell_field(self, state, *, quadrature, name=None):
        """Sample the piecewise-constant reconstruction of cell averages.

        This explicit numerical map preserves the native layout and exact stage.
        It does not publish a State or assert that arbitrary means equal points.
        """
        from pops.fields.spatial_interaction import CellMidpoint
        from pops.model.spaces import StateSpace
        from pops.time.values import _Coeff
        from pops.time.stencil import StencilAccess
        state = _resolve_handle(state)
        validate_input_regions(self, (state,), self._current_region(), "cell midpoint reconstruction")
        if type(quadrature) is not CellMidpoint or state.vtype != "state" or type(state.space) is not StateSpace:
            raise TypeError("cell_field requires an exact State and explicit CellMidpoint()")
        space = FieldSpace(name or "cell_midpoint", components=state.space.components,
            layout=state.space.layout, representation="field", centering=state.space.centering,
            units=state.space.units, frame=state.space.frame, clock=state.space.clock,
            support=state.space.support, sampling="cell_center", value_shape=state.space.value_shape,
            representation_data=state.space.representation_data)
        value = self._new("scalar_field", "linear_combine", (state,),
            {"coeffs": (_Coeff({0: 1}).to_polynomial(),), "ncomp": len(space.components),
             "cell_field_projection": "pops.cell-midpoint-sampling@1",
             "stencil_access": StencilAccess.pointwise()}, name, state.block,
            space=space, point=state.point, field_context=state.field_context,
            inherit_state_ref=False)
        validate_cell_field(value)
        return value

    @atomic_authoring
    def cell_mean(self, field, *, template, quadrature, name=None):
        """Integrate a computed cell field with the declared midpoint quadrature.

        This map is a numerical approximation, not an identity between sampling
        representations. Both reads stay provisional inside the same attempt.
        """
        from pops.fields.spatial_interaction import CellMidpoint
        from pops.time.expressions import encode_expressions
        field, template = _resolve_handle(field), _resolve_handle(template)
        validate_input_regions(self, (field, template), self._current_region(), "cell midpoint quadrature")
        if type(quadrature) is not CellMidpoint:
            raise TypeError("cell_mean requires an explicit CellMidpoint() quadrature")
        _midpoint_spaces(field, template)
        encoded, nodes, _ = encode_expressions(
            tuple(field[i] for i in range(len(field.space.components))), self)
        from pops.time.field_context import merge_field_provenance
        context = merge_field_provenance(field.field_context, template.field_context)
        value = self._new("state", "pointwise_expression", (field, template),
            {"expressions": encoded, "expression_nodes": nodes,
             "cell_mean_projection": "pops.cell-midpoint-quadrature@1"},
            name, template.block, space=template.space, point=template.point,
            state_ref=template.state_ref, field_context=context)
        validate_cell_mean(value)
        return value

    @atomic_authoring
    def spatial_interaction(self, state, kernel, *, output_space, measure, quadrature, realization, components=None, source_scope="issued", name=None, owner_block=None):
        """I_c(x)=sum_y W(x,y) rho_c(y) kappa_y volume_y, without commit.

        The result is an owner-qualified scalar field, not a physical State with
        invented units. Acceptance and observation/publication remain explicit.
        """
        self._guard_mutable("spatial interaction")
        from pops.fields.spatial_interaction import (
            SpatialInteractionKernel, CellVolumeMeasure, CellMidpoint, DirectSpatialInteraction,
        )
        state = _resolve_handle(state)
        if isinstance(state, ProgramValue) and state.op == "field_component" and (source_scope == "completed_original" or owner_block is not None):
            require_top_level(self, state, "closed original spatial interaction")
            selected = (0,) if components is None else tuple(components)
            if source_scope != "completed_original" or selected != (0,) or any(type(component) is not int for component in selected):
                raise ValueError("global field interaction requires explicit completed_original scope and scalar selection")
            if (type(kernel) is not SpatialInteractionKernel or type(measure) is not CellVolumeMeasure
                    or type(quadrature) is not CellMidpoint or type(realization) is not DirectSpatialInteraction):
                raise TypeError("closed interaction requires explicit kernel/measure/quadrature/realization")
            metadata = _closed_source_contract(self, state, owner_block, output_space)
            attrs = {"contract":"pops.spatial-interaction@3", "kernel":kernel.to_data(),
                "measure":measure.to_data(), "quadrature":"pops.cell-midpoint@1",
                "realization":"pops.direct-spatial-interaction@1", "ncomp":1, "components":(0,),
                "source_scope":"completed_original", "max_workspace_bytes":{"uint64_hex":"%016x" % realization.max_workspace_bytes},
                "closed_field_source":metadata}
            value = self._new("scalar_field", "spatial_interaction", (state,), attrs,
                              name, None, point=state.point, space=output_space)
            from types import MappingProxyType
            from .global_history_storage import _IssuedStorage, _ISSUER
            records = dict(getattr(self, "_closed_field_interaction_issuance", {}))
            records[value.id] = (value, _IssuedStorage(metadata, issuer=_ISSUER))
            object.__setattr__(self, "_closed_field_interaction_issuance", MappingProxyType(records))
            return value
        if owner_block is not None:
            raise ValueError("owner_block is storage-only for a completed global original field")
        if not isinstance(state, ProgramValue) or state.vtype != "state" or state.space is None:
            raise TypeError("spatial interaction requires a typed State")
        require_top_level(self, state, "spatial interaction")
        _source_point(state)
        if (type(kernel) is not SpatialInteractionKernel or type(measure) is not CellVolumeMeasure
                or type(quadrature) is not CellMidpoint or type(realization) is not DirectSpatialInteraction):
            raise TypeError("spatial interaction requires explicit physical kernel/measure/quadrature/direct realization")
        if source_scope not in ("issued", "accepted") or (source_scope == "accepted" and state.op != "state"):
            raise ValueError("accepted composite interaction requires the issued State.n carrier")
        if state.op == "history" and "history_contract" not in state.attrs:
            raise ValueError("composite interaction history requires an authenticated sample/point contract")
        width = len(state.space.components)
        indices = tuple(range(width)) if components is None else tuple(components)
        if not indices or len(set(indices)) != len(indices) or any(type(c) is not int or not 0 <= c < width for c in indices):
            raise ValueError("spatial interaction component selection is invalid")
        _spaces(state, output_space, indices)
        _units(state, output_space, indices, kernel.units, measure.coordinate_units, kernel.dimension)
        history_source = None
        if state.op == "history":
            from .spatial_history_source import history_source_contract
            history_source = history_source_contract(state)
        attrs = {
            "contract": "pops.spatial-interaction@2" if history_source else "pops.spatial-interaction@1", "kernel": kernel.to_data(),
            "measure": measure.to_data(), "quadrature": "pops.cell-midpoint@1",
            "realization": "pops.direct-spatial-interaction@1", "ncomp": len(indices),
            "components": indices, "source_scope": source_scope, "max_workspace_bytes": {"uint64_hex": "%016x" % realization.max_workspace_bytes},
        }
        if history_source is not None:
            attrs["history_source"] = history_source
        return self._new("scalar_field", "spatial_interaction", (state,), attrs,
                         name, state.block, point=state.point, space=output_space)
