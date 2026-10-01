"""One explicit nonlocal spatial map; publication belongs to the surrounding Program."""
from pops.time.values import ProgramValue, _resolve_handle
from pops.time._program.value_validation import require_top_level
from pops.time._authoring import atomic_authoring
from pops.model.spaces import FieldSpace
from pops._ir.quantity import PhysicalDimension


def _dimension(data):
    if data is None:
        return None
    from pops.time.canonical_data import _json_ready
    return PhysicalDimension.from_data(_json_ready(data))


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


def interaction_contract(value):
    from pops.fields.spatial_interaction import kernel_cpp
    if value.op != "spatial_interaction" or value.vtype != "scalar_field" or len(value.inputs) != 1:
        raise ValueError("invalid spatial interaction SSA contract")
    attrs = value.attrs
    required = {"contract", "kernel", "measure", "quadrature", "realization", "max_workspace_bytes", "components", "ncomp", "source_scope"}
    if set(attrs) != required or attrs["contract"] != "pops.spatial-interaction@1":
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
    _units(source, value.space, components, _dimension(kernel["units"]), tuple(_dimension(unit) for unit in measure["coordinate_units"]), dimension)
    return source, dimension, cpp, int(raw, 16), components


class _ProgramSpatialInteraction:
    @atomic_authoring
    def spatial_interaction(self, state, kernel, *, output_space, measure, quadrature, realization, components=None, source_scope="issued", name=None):
        """I_c(x)=sum_y W(x,y) rho_c(y) kappa_y volume_y, without commit.

        The result is an owner-qualified scalar field, not a physical State with
        invented units. Acceptance and observation/publication remain explicit.
        """
        self._guard_mutable("spatial interaction")
        from pops.fields.spatial_interaction import (
            SpatialInteractionKernel, CellVolumeMeasure, CellMidpoint, DirectSpatialInteraction,
        )
        state = _resolve_handle(state)
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
        return self._new("scalar_field", "spatial_interaction", (state,), {
            "contract": "pops.spatial-interaction@1", "kernel": kernel.to_data(),
            "measure": measure.to_data(), "quadrature": "pops.cell-midpoint@1",
            "realization": "pops.direct-spatial-interaction@1", "ncomp": len(indices),
            "components": indices, "source_scope": source_scope, "max_workspace_bytes": {"uint64_hex": "%016x" % realization.max_workspace_bytes},
        }, name, state.block, point=state.point, space=output_space)
