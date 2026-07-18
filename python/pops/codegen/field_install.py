"""Generic orchestration of prepared field-method, solver and nullspace providers."""
from __future__ import annotations

from typing import Any, NoReturn

from pops.codegen.field_install_plan import ResolvedFieldInstallPlan, native_plain_data
from pops.codegen.lowering_coverage import (
    LoweringCoverageReport,
    LoweringCoverageRow,
    LoweringRejection,
)
from pops.fields._identity import field_identity, strict_field_data
from pops.fields._prepared_field_lowering_registry import (
    PreparedFieldLoweringRequest,
    prepared_field_lowering_binding_from_descriptor,
)
from pops.fields._prepared_field_nullspace_registry import (
    prepared_field_nullspace_binding,
)
from pops.fields._prepared_field_solver_registry import (
    prepared_field_solver_binding_from_descriptor,
)
from pops.fields.discretization import (
    field_discretization_data,
    require_field_discretization,
)
from pops.fields.operator import FieldOperator
from pops.identity import Identity, canonical_bytes


def _reject(
    rows: list[LoweringCoverageRow], source: str, gate: str, message: str,
) -> NoReturn:
    report = LoweringCoverageReport((*rows, LoweringCoverageRow(
        source, "rejected", gate=gate
    )))
    raise LoweringRejection(
        message, coverage_report=report, source=source, gate=gate
    )


def resolve_field_install_plan(
    name: str,
    registration: Any,
    *,
    target: str,
    rhs_providers: tuple[Any, ...],
    provider_route: tuple[dict[str, Any], ...],
    output_components: tuple[str, ...],
    layout: Any,
) -> ResolvedFieldInstallPlan:
    """Authenticate and combine providers without interpreting spatial semantics."""
    operator = getattr(registration, "operator", None)
    plan = getattr(registration, "discretization", None)
    if not isinstance(operator, FieldOperator):
        raise TypeError("field registration lost its FieldOperator")
    plan = require_field_discretization(
        plan, where="field registration discretization"
    )
    if type(target) is not str or not target:
        raise TypeError("field installation target must be a non-empty exact identity")
    if not rhs_providers or len(rhs_providers) != len(provider_route):
        _reject(
            [],
            "field:%s:provider" % name,
            "field.provider.pack_invalid",
            "field %r requires a non-empty, fully routed provider pack" % name,
        )
    provider_identity = {
        "schema_version": 1,
        "contributions": [
            {
                "provider": provider.canonical_identity(),
                "owner_block": route["owner_block"],
                "native_key": route["key"],
                "coefficient": route["coefficient"],
                "measure": route["measure"],
            }
            for provider, route in zip(rhs_providers, provider_route, strict=True)
        ],
    }
    provider_slot = field_identity(
        "qualified-field-provider", provider_identity
    ).token
    request = PreparedFieldLoweringRequest(
        name,
        operator,
        plan,
        target,
        tuple(output_components),
        layout,
    )
    try:
        lowering_binding = prepared_field_lowering_binding_from_descriptor(
            plan.method, request=request, where="field %r method" % name
        )
    except LoweringRejection:
        raise
    except (NotImplementedError, TypeError, ValueError) as exc:
        _reject(
            [],
            "field:%s:method" % name,
            "field.method.provider_incompatible",
            str(exc),
        )
    resolution = lowering_binding.resolution
    if resolution.solver_facts.target != target:
        raise ValueError("field lowering provider changed the requested target identity")
    rows = [
        LoweringCoverageRow.from_data(evidence.to_data())
        for evidence in resolution.evidence
    ]
    rows.append(LoweringCoverageRow(
        "field:%s:provider" % name,
        "lowered",
        ("field-install:%s:provider:%s" % (name, provider_slot),),
    ))
    for reference in operator.declaration_references():
        rows.append(LoweringCoverageRow(
            "field:%s:dependency:%s" % (name, reference.qualified_id),
            "lowered",
            ("field-install:%s:qualified-dependency" % name,),
        ))

    try:
        solver_binding = prepared_field_solver_binding_from_descriptor(
            plan.solver,
            facts=resolution.solver_facts,
            where="field %r solver" % name,
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        _reject(
            rows,
            "field:%s:solver" % name,
            "field.solver.provider_incompatible",
            str(exc),
        )
    rows.append(LoweringCoverageRow(
        "field:%s:solver" % name,
        "lowered",
        ("field-install:%s:solver-provider:%s" % (
            name, solver_binding.identity
        ),),
    ))

    nonlinear_options = None
    nonlinear_provider = None
    if plan.nonlinear is not None:
        nonlinear_adapter = getattr(plan.nonlinear, "lower_field_nonlinear", None)
        if not callable(nonlinear_adapter):
            _reject(
                rows,
                "field:%s:nonlinear" % name,
                "field.nonlinear.not_native",
                "field %r nonlinear solver has no lowering adapter" % name,
            )
        try:
            nonlinear_provider = nonlinear_adapter(target=target, layout=layout)
        except (TypeError, ValueError) as exc:
            _reject(
                rows,
                "field:%s:nonlinear" % name,
                "field.nonlinear.layout_incompatible",
                str(exc),
            )
        manifest = getattr(nonlinear_provider, "to_data", None)
        install_adapter = getattr(nonlinear_provider, "install", None)
        capabilities = getattr(nonlinear_provider, "capabilities", frozenset())
        identity = getattr(nonlinear_provider, "identity", None)
        if (
            not callable(manifest)
            or not callable(install_adapter)
            or not isinstance(identity, Identity)
            or not {"residual", "publication_atomic", "reject_attempt"}.issubset(
                set(capabilities)
            )
        ):
            _reject(
                rows,
                "field:%s:nonlinear" % name,
                "field.nonlinear.invalid_adapter",
                "field %r nonlinear solver returned an invalid prepared provider" % name,
            )
        nonlinear_options = manifest()
        rows.append(LoweringCoverageRow(
            "field:%s:nonlinear" % name,
            "lowered",
            ("field-install:%s:nonlinear:%s" % (name, identity.token),),
        ))

    try:
        nullspace_binding = prepared_field_nullspace_binding(
            plan.nullspace,
            plan.gauge,
            facts=resolution.nullspace_facts,
            where="field %r" % name,
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        _reject(
            rows,
            "field:%s:nullspace" % name,
            "field.nullspace.provider_incompatible",
            str(exc),
        )
    rows.append(LoweringCoverageRow(
        "field:%s:nullspace" % name,
        "lowered" if nullspace_binding.resolution.singular else "documentary",
        (("field-install:%s:nullspace-provider:%s" % (
            name, nullspace_binding.identity
        ),) if nullspace_binding.resolution.singular else ()),
    ))
    rows.append(LoweringCoverageRow(
        "field:%s:gauge" % name,
        "lowered" if nullspace_binding.resolution.singular else "documentary",
        (("field-install:%s:gauge-provider:%s" % (
            name, nullspace_binding.identity
        ),) if nullspace_binding.resolution.singular else ()),
    ))

    options = {
        **native_plain_data(resolution.native_options),
        "provider_slot": provider_slot,
        "provider_identity": provider_identity,
        "provider_identity_text": canonical_bytes(
            strict_field_data(provider_identity)
        ).hex(),
        "provider_pack": [dict(route) for route in provider_route],
        "method_provider": lowering_binding.to_data(),
        "solver_provider": solver_binding.to_data(),
        "nullspace_provider": nullspace_binding.to_data(),
        "nonlinear": nonlinear_options,
    }
    report = LoweringCoverageReport(rows)
    data = {
        "schema_version": 1,
        "name": name,
        "operator": operator.to_data(),
        "discretization": field_discretization_data(
            plan, where="resolved field install discretization"
        ),
        "target": target,
        "rhs_providers": [
            provider.canonical_identity() for provider in rhs_providers
        ],
        "native_options": options,
        "coverage": report.to_data(),
    }
    identity = field_identity("resolved-field-install", data)
    return ResolvedFieldInstallPlan(
        name,
        operator,
        plan,
        target,
        rhs_providers,
        options,
        report,
        nonlinear_provider,
        identity,
    )


__all__ = ["ResolvedFieldInstallPlan", "resolve_field_install_plan"]
