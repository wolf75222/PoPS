"""Prepared native shared-constant kernel for a joint field problem."""
from __future__ import annotations

import json
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


def _author(options: Any, gauge: Any, _properties: Any, where: str) -> Any:
    if set(options) != {"components"} or type(options["components"]) is not int \
            or options["components"] != 2:
        raise TypeError("%s shared-constant provider currently requires exactly two components" % where)
    if type(gauge) is not SharedMeanGauge or len(gauge.unknowns) != options["components"]:
        raise TypeError("%s requires one SharedMeanGauge over the exact field tuple" % where)
    return PreparedNullspaceContracts(
        {"basis": "shared-constant-vector", "basis_count": 1, "components": 2},
        {"constraint": "mean-sum", "value": scalar_literal(gauge.value)},
    )


def _validate(use: PreparedNullspaceUse, where: str) -> None:
    nullspace, gauge = use.contracts.detached()
    if nullspace != {"basis": "shared-constant-vector", "basis_count": 1, "components": 2}:
        raise ValueError("%s has an invalid joint constant basis" % where)
    if set(gauge) != {"constraint", "value"} or gauge["constraint"] != "mean-sum" \
            or type(gauge["value"]) is not ScalarLiteral:
        raise ValueError("%s has an invalid joint gauge" % where)
    if use.components is not None and use.components != 2:
        raise ValueError("%s joint field kernel requires exactly two packed components" % where)
    if not use.operator_properties["symmetric"] or use.operator_properties["positive_definite"]:
        raise ValueError("%s shared kernel requires symmetry and positivity on its complement" % where)


def _emit(_node: Any, _prelude: Any, contracts: Any, identity: str, _provider: Any) -> Any:
    expression = (
        "[&]() { auto plan = pops::constant_mean_zero_nullspace<pops::kNativeDimension>(%s, "
        "\"one shared field constant mode\"); plan.bases.front().component_count = 2; "
        "plan.gauges.front().value = static_cast<pops::Real>(%s) / pops::Real(2); "
        "return plan; }()"
        % (json.dumps(identity), scalar_cpp(contracts.gauge["value"]))
    )
    return PreparedNullspaceNativeEmission(expression)


_PROVIDER = register_prepared_nullspace_provider(PreparedNullspaceProvider(
    provider_id="pops.prepared-nullspace.shared-field-constant",
    emitter_id="pops.prepared-nullspace.shared-field-constant@1",
    singular=True,
    use_policy=PreparedNullspaceUsePolicy(
        "pops.prepared-nullspace.shared-field-constant-use", 1,
        {"components": {"minimum": 2, "maximum": 2}, "basis_count": 1, "gauge": "mean-sum"},
        _validate),
    author=_author,
    emitter=_emit,
    native_component=PreparedNativeComponent.pops_builtin(
        "pops.prepared-nullspace.shared-field-constant",
        entry_headers=("pops/numerics/elliptic/interface/field_nullspace.hpp",
                       "pops/numerics/elliptic/interface/field_nullspace_workspace.hpp")),
))


def shared_constant_nullspace() -> PreparedNullspace:
    return PreparedNullspace(_PROVIDER, components=2)
