"""Lower authored constant vectors into the generic prepared nullspace mask contract."""
from __future__ import annotations

import json
from typing import Any

from pops.identity.scalar import ScalarLiteral, scalar_cpp, scalar_literal
from pops.native_components import PreparedNativeComponent
from ._field_matrix import gauge_modes, inverse
from ._program_expression import decode_field_literal
from ._prepared_nullspace_registry import (
    PreparedNullspaceContracts, PreparedNullspaceNativeEmission,
    PreparedNullspaceProvider, PreparedNullspaceUsePolicy, register_prepared_nullspace_provider,
)
from .nullspace import PreparedNullspace
from .problem import ConstantModeGauge


def _author(options, gauge, operator_properties, where):
    if type(gauge) is not ConstantModeGauge or options != {"components": len(gauge.unknowns)}:
        raise TypeError("%s requires the exact constant-mode gauge tuple" % where)
    modes, values = gauge_modes(gauge, len(gauge.unknowns))
    gram = tuple(tuple(sum(a*b for a, b in zip(left, right)) for right in modes) for left in modes)
    inverse_gram = inverse(gram, where="constant modes are linearly dependent")
    coordinates = tuple(sum(a*b for a, b in zip(row, values)) for row in inverse_gram)
    return PreparedNullspaceContracts(
        {"components": len(gauge.unknowns), "modes": tuple(tuple(scalar_literal(x).to_data() for x in row) for row in modes)},
        {"coordinates": tuple(scalar_literal(x) for x in coordinates)})


def _validate(use, where):
    basis, gauge = use.contracts.detached()
    n = basis.get("components")
    modes = basis.get("modes")
    if set(basis) != {"components", "modes"} or type(n) is not int or n < 1 or not isinstance(modes, tuple) or not modes:
        raise ValueError("%s has malformed constant modes" % where)
    if any(not isinstance(row, tuple) or len(row) != n for row in modes):
        raise ValueError("%s constant mode width differs from the field tuple" % where)
    for row in modes:
        for x in row:
            decode_field_literal(x)
    if set(gauge) != {"coordinates"} or len(gauge["coordinates"]) != len(modes) or any(type(x) is not ScalarLiteral for x in gauge["coordinates"]):
        raise ValueError("%s has malformed constant mode constraints" % where)
    if use.components is not None and use.components != n:
        raise ValueError("%s changes constant mode component authority" % where)
    if not use.operator_properties["symmetric"] or not use.operator_properties["positive_definite_on_nullspace_complement"]:
        raise ValueError("%s requires a symmetric positive operator on the kernel complement" % where)


def _emit(node, prelude, contracts, plan_identity, provider):
    n = contracts.nullspace["components"]
    lines = ["[&]() { pops::FieldNullspacePlan<pops::kNativeDimension> plan;",
             "plan.identity = %s; plan.layout_identity = %s;" % (json.dumps(plan_identity), json.dumps(plan_identity + ":layout"))]
    for i, (mode, coordinate) in enumerate(zip(contracts.nullspace["modes"], contracts.gauge["coordinates"])):
        lines.extend(["{ auto mask = std::make_shared<pops::MultiFab<pops::kNativeDimension>>(ctx.alloc_scalar_field(%d, 0));" % n,
            "for (std::size_t li = 0; li < mask->local_size(); ++li) { auto view = mask->fab(li).view();",
            "pops::for_each_cell(mask->box(li), [=] POPS_HD(const pops::Index<pops::kNativeDimension>& cell) {"])
        lines.extend("view(cell, %d) = static_cast<pops::Real>(%s);" % (j, scalar_cpp(decode_field_literal(x))) for j, x in enumerate(mode))
        lines.extend(["}); } pops::FieldNullspaceBasis<pops::kNativeDimension> basis;",
            "basis.identity = %s; basis.recipe_identity = basis.identity; basis.provenance = \"authored constant field mode\";" % json.dumps(plan_identity + ":mode:" + str(i)),
            "basis.component_count = %d; basis.masks.push_back(mask);" % n,
            "plan.gauges.push_back(pops::FieldGaugeConstraint{basis.identity, static_cast<pops::Real>(%s)});" % scalar_cpp(coordinate),
            "plan.bases.push_back(std::move(basis)); }"])
    lines.append("return plan; }()")
    return PreparedNullspaceNativeEmission(" ".join(lines))


_PROVIDER = register_prepared_nullspace_provider(PreparedNullspaceProvider(
    provider_id="pops.prepared-nullspace.constant-field-modes",
    emitter_id="pops.prepared-nullspace.constant-field-modes@1", singular=True,
    use_policy=PreparedNullspaceUsePolicy("pops.prepared-nullspace.constant-field-modes-use", 1,
        {"components": {"minimum": 1}, "basis": "authored-constant-vectors"}, _validate),
    author=_author, emitter=_emit,
    native_component=PreparedNativeComponent.pops_builtin("pops.prepared-nullspace.constant-field-modes",
        entry_headers=("pops/numerics/elliptic/interface/field_nullspace.hpp",
                       "pops/numerics/elliptic/interface/field_nullspace_workspace.hpp"))))


def constant_mode_nullspace(gauge: ConstantModeGauge) -> PreparedNullspace:
    return PreparedNullspace(_PROVIDER, components=len(gauge.unknowns))
