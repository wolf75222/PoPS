"""Read-only builders for model and program manifest rows."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._module_manifest import ModuleManifest
from ._operator_manifest import OperatorManifestEntry, OperatorRegistryManifest
from .manifest_support import (
    field_space_row as _field_space_row,
    native_catalog as _native_catalog,
    native_routes as _native_routes,
    param_row as _param_row,
    params_utilization as _params_utilization,
    state_space_row as _state_space_row,
)
from .ownership import OwnerPath
from .provider_pack import build_provider_pack


def _declaration_row(
    descriptor: Any,
    row: Any,
    *,
    handle: Any,
    index: Any,
    name: str,
    kind: str,
    owner: OwnerPath,
    where: str,
) -> dict[str, Any]:
    if getattr(descriptor, "name", None) != name:
        raise ValueError(
            "%s registry key %r does not match descriptor name %r"
            % (where, name, getattr(descriptor, "name", None))
        )
    if not isinstance(row, Mapping):
        raise TypeError("%s row %r must be a mapping" % (where, name))
    resolved = index.authenticate(handle)._resolved(owner)
    if resolved.local_id != name or resolved.kind != kind:
        raise ValueError("%s registry handle does not match %s:%s" % (where, kind, name))
    result = dict(row)
    result["qid"] = resolved.qualified_id
    result["handle"] = resolved.canonical_identity()
    return result


def _operator_alias_rows(
    module: Any,
    registry: Any,
    index: Any,
    *,
    owner: OwnerPath,
) -> dict[str, Any]:
    rows = {}
    for alias, target in sorted(registry.aliases().items()):
        alias_handle = index.authenticate(module.operator_handle(alias))._resolved(owner)
        target_handle = index.authenticate(module.operator_handle(target))._resolved(owner)
        rows[alias] = {
            "name": alias,
            "target": target,
            "qid": alias_handle.qualified_id,
            "handle": alias_handle.canonical_identity(),
            "target_qid": target_handle.qualified_id,
            "target_handle": target_handle.canonical_identity(),
        }
    return rows


def _operator_binding_rows(
    module: Any,
    index: Any,
    *,
    owner: OwnerPath,
) -> list[dict[str, Any]]:
    """Canonical rows for typed scientific-handle -> operator projections."""
    rows = []
    bindings = module.operator_bindings()
    ordered = sorted(
        bindings.items(),
        key=lambda item: (
            item[0].kind,
            item[0].local_id,
            item[0].schema_version,
            item[1].registered_operator_name,
        ),
    )
    for subject, target in ordered:
        resolved_subject = index.authenticate(subject)._resolved(owner)
        resolved_target = index.authenticate(target)._resolved(owner)
        rows.append({
            "subject_qid": resolved_subject.qualified_id,
            "subject_handle": resolved_subject.canonical_identity(),
            "target_qid": resolved_target.qualified_id,
            "target_handle": resolved_target.canonical_identity(),
        })
    return rows


def build_module_manifest(module: Any) -> ModuleManifest:
    """Build a canonical ModuleManifest through authoritative public registries."""
    module_owner = OwnerPath.coerce(module)
    canonical_owner = module_owner.canonical()
    registry = module.operator_registry()
    if registry.owner_path != module_owner:
        raise ValueError("Module operator registry is not owned by the Module authority")
    index = module.declaration_index()
    if index.owner_path != module_owner:
        raise ValueError("Module operator declaration index has a different owner authority")
    entries = []
    for operator in registry:
        resolved = index.authenticate(module.operator_handle(operator.name))._resolved(
            canonical_owner
        )
        entries.append(
            OperatorManifestEntry(operator, registry.id_of(operator.name), resolved)
        )
    aliases = _operator_alias_rows(module, registry, index, owner=canonical_owner)
    operators = OperatorRegistryManifest(entries, aliases=aliases, owner=canonical_owner)
    operator_bindings = _operator_binding_rows(module, index, owner=canonical_owner)
    declared_states = module.state_spaces()
    declared_fields = module.field_spaces()
    declared_params = module.params()
    declared_aux = module.aux()
    state_spaces = {
        name: _declaration_row(
            descriptor,
            _state_space_row(descriptor),
            handle=module.state_handle(descriptor),
            index=index,
            name=name,
            kind="state",
            owner=canonical_owner,
            where="state space",
        )
        for name, descriptor in sorted(declared_states.items())
    }
    field_spaces = {
        name: _declaration_row(
            descriptor,
            _field_space_row(descriptor),
            handle=module.field_handle(descriptor),
            index=index,
            name=name,
            kind="field",
            owner=canonical_owner,
            where="field space",
        )
        for name, descriptor in sorted(declared_fields.items())
    }
    params = {
        name: _declaration_row(
            descriptor,
            _param_row(descriptor),
            handle=module.param_handle(descriptor),
            index=index,
            name=name,
            kind="parameter",
            owner=canonical_owner,
            where="parameter",
        )
        for name, descriptor in sorted(declared_params.items())
    }
    aux = {
        name: _declaration_row(
            descriptor,
            {
                "aux_kind": getattr(descriptor, "kind", "cell_scalar"),
                "representation": descriptor.representation,
                "centering": descriptor.centering,
                "unit": descriptor.unit,
                "frame": descriptor.frame,
                "clock": descriptor.clock,
            },
            handle=module.aux_handle(descriptor),
            index=index,
            name=name,
            kind="aux",
            owner=canonical_owner,
            where="aux field",
        )
        for name, descriptor in sorted(declared_aux.items())
    }
    eigenvalues = getattr(module, "_eigenvalues", None)
    has_eigenvalues = {
        "x": bool(eigenvalues and eigenvalues.get("x")),
        "y": bool(eigenvalues and eigenvalues.get("y")),
    }
    wave_speed_provider = getattr(module, "wave_speed_provider_kind", None)
    capabilities_provider = getattr(module, "capabilities", None)
    raw_capabilities = capabilities_provider() if callable(capabilities_provider) else {}
    if not isinstance(raw_capabilities, Mapping):
        raise TypeError("Module capabilities() must return a mapping")
    capabilities = dict(raw_capabilities)
    routes = _native_routes()
    catalog = _native_catalog()
    provider_pack = build_provider_pack(module).to_data()
    return ModuleManifest(
        name=module.name,
        owner_path=canonical_owner,
        state_spaces=state_spaces,
        field_spaces=field_spaces,
        params=params,
        aux=aux,
        provider_pack=provider_pack,
        has_eigenvalues=has_eigenvalues,
        wave_speed_provider=wave_speed_provider,
        operators=operators,
        operator_bindings=operator_bindings,
        capabilities=capabilities,
        native_routes=routes,
        native_catalog=catalog,
        abi_requirements={"route_registry_signature": routes["signature"], "abi_key": None},
        params_utilization=_params_utilization(params),
    )


_CONDENSED_ROUTE_OPS = ("condensed_coeffs", "condensed_rhs", "condensed_reconstruct")


def condensed_route_manifest(program: Any) -> Any:
    values = getattr(program, "_values", None)
    if values is None:
        return None
    present = [value.op for value in values if value.op in _CONDENSED_ROUTE_OPS]
    if not present:
        return None
    seen = []
    for operation in present:
        if operation not in seen:
            seen.append(operation)
    return {
        "route": "condensed_implicit",
        "ops": seen,
        "operator_module": "pops::detail (block_inverse)",
        "operator_header": "pops/numerics/linalg/block_inverse.hpp",
        "authoring": "pops.Program.solve",
        "limitations": {
            "bit_exact_theta": 1.0,
            "bit_exact_scope": "first_step",
            "theta_range": "(0, 1]",
            "note": (
                "matrix-free BiCGStab on a uniform layout and a composite tensor provider on a "
                "refined hierarchy; theta < 1 carries phi through Program history"
            ),
        },
    }


def coupling_operator_manifest(
    compiled: Any,
    conserved: Any = (),
    created: Any = (),
    frequency: Any = None,
) -> Any:
    from pops._ir.literals import scalar_data

    conserved, created = list(conserved), list(created)
    freq = getattr(compiled, "frequency", 0.0) if frequency is None else frequency
    per_cell = bool(
        getattr(compiled, "freq_prog_ops", []) or getattr(compiled, "freq_prog_args", [])
    )
    return {
        "route": "coupled_source",
        "name": getattr(compiled, "name", "coupled_source"),
        "operator_module": "pops::CouplingOperator",
        "operator_header": "pops/coupling/source/coupling_operator.hpp",
        "in_fields": list(
            zip(
                getattr(compiled, "in_blocks", []),
                getattr(compiled, "in_roles", []),
                strict=True,
            )
        ),
        "out_terms": list(
            zip(
                getattr(compiled, "out_blocks", []),
                getattr(compiled, "out_roles", []),
                strict=True,
            )
        ),
        "conservation": {
            "conserved_roles": conserved,
            "created_roles": created,
            "unchecked": not (conserved or created),
        },
        "frequency": {"constant_mu": scalar_data(freq), "per_cell": per_cell},
        "utilization": compiled.utilization() if hasattr(compiled, "utilization") else None,
    }


def _is_manifestable_module(obj: Any) -> bool:
    return obj is not None and hasattr(obj, "operator_registry") and hasattr(obj, "state_spaces")


def module_manifest_of(model_or_module: Any) -> Any:
    module = model_or_module
    if not _is_manifestable_module(module):
        module = getattr(module, "module", None)
    if not _is_manifestable_module(module):
        return None
    return build_module_manifest(module)


__all__ = [
    "build_module_manifest",
    "condensed_route_manifest",
    "coupling_operator_manifest",
    "module_manifest_of",
]
