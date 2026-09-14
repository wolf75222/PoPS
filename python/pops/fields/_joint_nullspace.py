"""Prepared native shared-constant kernel for a joint field problem."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pops.identity.scalar import ScalarLiteral, scalar_cpp, scalar_literal
from pops.native_components import PreparedNativeComponent

from ._prepared_nullspace_registry import (
    PreparedNullspaceContracts,
    PreparedNullspaceNativeEmission,
    PreparedNullspaceProvider,
    PreparedNullspaceUse,
    PreparedNullspaceUsePolicy,
    register_prepared_nullspace_provider,
)
from .nullspace import PreparedNullspace
from .problem import SharedMeanGauge


def _author(options: Mapping[str, Any], gauge: Any, operator_properties: Mapping[str, bool],
            where: str) -> PreparedNullspaceContracts:
    if set(options) != {"components"} or type(options["components"]) is not int \
            or options["components"] < 1:
        raise TypeError("%s shared-constant provider requires a positive component count" % where)
    if type(gauge) is not SharedMeanGauge or len(gauge.unknowns) != options["components"]:
        raise TypeError("%s requires one SharedMeanGauge over the exact field tuple" % where)
    return PreparedNullspaceContracts(
        {"basis": "shared-constant-vector", "basis_count": 1, "components": options["components"]},
        {"constraint": "mean-sum", "value": scalar_literal(gauge.value)},
    )


def _validate(use: PreparedNullspaceUse, where: str) -> None:
    nullspace, gauge = use.contracts.detached()
    components = nullspace.get("components")
    if type(components) is not int or components < 1 or nullspace != {"basis": "shared-constant-vector", "basis_count": 1, "components": components}:
        raise ValueError("%s has an invalid joint constant basis" % where)
    if set(gauge) != {"constraint", "value"} or gauge["constraint"] != "mean-sum" \
            or type(gauge["value"]) is not ScalarLiteral:
        raise ValueError("%s has an invalid joint gauge" % where)
    if use.components is not None and use.components != components:
        raise ValueError("%s joint field kernel changes its exact packed component count" % where)
    if not use.operator_properties["symmetric"] or not use.operator_properties["positive_definite_on_nullspace_complement"]:
        raise ValueError("%s shared kernel requires symmetry and positivity on its complement" % where)


def _emit(node: Any, prelude: list[str], contracts: PreparedNullspaceContracts,
          plan_identity: str, provider: PreparedNullspaceProvider) -> PreparedNullspaceNativeEmission:
    expression = (
        "[&]() { auto plan = pops::constant_mean_zero_nullspace<pops::kNativeDimension>(%s, "
        "\"one shared field constant mode\"); plan.bases.front().component_count = %d; "
        "plan.gauges.front().value = static_cast<pops::Real>(%s) / pops::Real(%d); "
        "return plan; }()"
        % (json.dumps(plan_identity), contracts.nullspace["components"], scalar_cpp(contracts.gauge["value"]), contracts.nullspace["components"])
    )
    return PreparedNullspaceNativeEmission(expression)


_PROVIDER = register_prepared_nullspace_provider(PreparedNullspaceProvider(
    provider_id="pops.prepared-nullspace.shared-field-constant",
    emitter_id="pops.prepared-nullspace.shared-field-constant@1",
    singular=True,
    use_policy=PreparedNullspaceUsePolicy(
        "pops.prepared-nullspace.shared-field-constant-use", 1,
        {"components": {"minimum": 1}, "basis_count": 1, "gauge": "mean-sum"},
        _validate),
    author=_author,
    emitter=_emit,
    native_component=PreparedNativeComponent.pops_builtin(
        "pops.prepared-nullspace.shared-field-constant",
        entry_headers=("pops/numerics/elliptic/interface/field_nullspace.hpp",
                       "pops/numerics/elliptic/interface/field_nullspace_workspace.hpp")),
))


def shared_constant_nullspace(components: int = 2) -> PreparedNullspace:
    return PreparedNullspace(_PROVIDER, components=components)


def constant_mode_nullspace(gauge: Any) -> PreparedNullspace:
    if type(gauge) is SharedMeanGauge:
        return shared_constant_nullspace(len(gauge.unknowns))
    from ._constant_mode_nullspace import constant_mode_nullspace as prepare
    return prepare(gauge)
