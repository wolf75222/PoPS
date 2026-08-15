"""C++ source emission for the sole native production package."""

from __future__ import annotations

import json
from collections.abc import Mapping
import math
from typing import Any, cast

# ---------------------------------------------------------------------------
# Backend / capability tables (single source of truth in this module)
# ---------------------------------------------------------------------------

_BACKEND_CAPS = {
    "production": {"cpu": True, "mpi": True, "amr": True, "gpu": False, "tier": "production"},
}


def compiled_capability_flags(backend: str) -> dict[str, bool]:
    """Project backend report metadata onto the bool-only compiled-model ABI contract."""
    try:
        row = _BACKEND_CAPS[backend]
    except KeyError:
        raise ValueError("unknown native backend %r" % backend) from None
    flags = {name: value for name, value in row.items() if type(value) is bool}
    if not flags:
        raise ValueError("native backend %r declares no compiled capability flags" % backend)
    return flags


def _normalize_native_amr_field_roles(value: Any) -> tuple[dict[str, Any], ...]:
    """Validate the resolved per-block field roles consumed by one AMR package.

    Output ownership and RHS contribution are deliberately separate rows.  The compiler receives
    these rows from the complete resolved Case; a model-local elliptic declaration is never used to
    guess which runtime field plan or block owns either role.
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError("native AMR field roles must be an ordered row sequence")
    try:
        rows = tuple(value)
    except TypeError:
        raise TypeError("native AMR field roles must be an ordered row sequence") from None
    result: list[dict[str, Any]] = []
    output_roles: set[str] = set()
    rhs_roles: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("native AMR field role %d must be a mapping" % index)
        kind = row.get("kind")
        field = row.get("field")
        block = row.get("block")
        if type(field) is not str or not field or type(block) is not str or not block:
            raise TypeError("native AMR field roles require non-empty field and block identities")
        if kind == "output":
            if set(row) != {"kind", "field", "block", "output_keys", "gradient_sign"}:
                raise TypeError("native AMR output role has an invalid exact shape")
            gradient_sign = row["gradient_sign"]
            if type(gradient_sign) is not int or gradient_sign not in (-1, 1):
                raise ValueError("native AMR output role requires gradient_sign=-1 or 1")
            keys = tuple(row["output_keys"])
            if not keys:
                raise ValueError("native AMR output role requires at least one exact output key")
            normalized_keys = []
            for key in keys:
                if (
                    not isinstance(key, Mapping)
                    or set(key)
                    != {
                        "owner_qid",
                        "space_kind",
                        "space_name",
                        "component",
                    }
                    or any(type(item) is not str or not item for item in key.values())
                ):
                    raise TypeError("native AMR output role carries an invalid ComponentKey")
                normalized_keys.append(
                    {
                        name: key[name]
                        for name in (
                            "owner_qid",
                            "space_kind",
                            "space_name",
                            "component",
                        )
                    }
                )
            if field in output_roles:
                raise ValueError("native AMR package has a duplicate field-output role")
            output_roles.add(field)
            result.append(
                {
                    "kind": "output",
                    "field": field,
                    "block": block,
                    "output_keys": tuple(normalized_keys),
                    "gradient_sign": gradient_sign,
                }
            )
            continue
        if kind == "rhs":
            if set(row) != {
                "kind",
                "field",
                "block",
                "binding_ordinal",
                "binding_identity",
                "provider_key",
                "coefficient",
            }:
                raise TypeError("native AMR RHS role has an invalid exact shape")
            ordinal = row["binding_ordinal"]
            binding_identity = row["binding_identity"]
            provider_key = row["provider_key"]
            coefficient = row["coefficient"]
            if type(ordinal) is not int or ordinal < 0:
                raise TypeError("native AMR RHS role requires a non-negative binding ordinal")
            if (
                type(binding_identity) is not str
                or not binding_identity
                or type(provider_key) is not str
                or not provider_key
            ):
                raise TypeError("native AMR RHS role requires binding and provider identities")
            if isinstance(coefficient, bool):
                raise TypeError("native AMR RHS coefficient must be a finite binary64 value")
            try:
                coefficient = float(coefficient)
            except (TypeError, ValueError, OverflowError):
                raise TypeError(
                    "native AMR RHS coefficient must be a finite binary64 value"
                ) from None
            if not math.isfinite(coefficient):
                raise ValueError("native AMR RHS coefficient must be finite")
            role = (field, ordinal)
            if role in rhs_roles:
                raise ValueError("native AMR package has a duplicate RHS binding role")
            rhs_roles.add(role)
            result.append(
                {
                    "kind": "rhs",
                    "field": field,
                    "block": block,
                    "binding_ordinal": ordinal,
                    "binding_identity": binding_identity,
                    "provider_key": provider_key,
                    "coefficient": coefficient,
                }
            )
            continue
        raise ValueError("native AMR field role kind must be 'output' or 'rhs'")
    return tuple(result)


def _native_amr_field_roles_identity(
    normalized_field_roles: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Project normalized runtime roles onto the float-free artifact-identity vocabulary."""
    return tuple(
        (
            {**role, "coefficient": role["coefficient"].hex()}
            if role["kind"] == "rhs"
            else dict(role)
        )
        for role in normalized_field_roles
    )


# ---------------------------------------------------------------------------
# model_hash -- stable hash of a HyperbolicModel
# ---------------------------------------------------------------------------


def model_hash(model: Any, params: Any = None) -> str:
    """Stable hash of *model* (a ``HyperbolicModel``): formulas
    (flux/eig/source/elliptic/primitives/cons_from) + roles + n_aux + gamma
    (+ any NAMED params). Single source of the hash, reused by
    ``Model._model_hash`` (which passes its Params). Serves to identify/reuse
    an already compiled .so (cache key) and to trace the run. Relies on
    ``repr(Expr)`` (stable, structural); insensitive to dict ordering (sorted).
    """
    import hashlib
    import json

    # Import the helper lazily to avoid pulling pops.dsl at import time.
    # aux_total_n_aux and roles_for live in dsl; we read them from the model
    # package which is stdlib-only (no C extension).
    from pops.identity.scalar import scalar_data
    from pops._ir.values import _EIG_FIELDS  # noqa: F401 -- confirm ir is importable
    from pops._cartesian_axes import canonical_axis_mapping

    def _scalar_token(value: Any) -> str:
        return json.dumps(scalar_data(value), sort_keys=True, separators=(",", ":"))

    def _axis_names(mapping: Any, *, where: str) -> tuple[str, ...]:
        if not mapping:
            return ()
        return tuple(canonical_axis_mapping(mapping, where=where))

    def _roles_for(names: Any, override: Any = None) -> list:
        from pops.physics.aux import roles_for

        return list(roles_for(names, override))

    m = model
    parts = []
    parts.append("name=%s" % m.name)
    parts.append("cons=%s" % ",".join(m.cons_names))
    parts.append("croles=%s" % ",".join(_roles_for(m.cons_names, m.cons_roles)))
    parts.append("prim_state=%s" % ",".join(m.prim_state))
    parts.append("proles=%s" % ",".join(_roles_for(m.prim_state, m.prim_roles)))
    parts.append("prim=%s" % ";".join("%s=%r" % (k, m.prim_defs[k]) for k in m.prim_defs))
    recovery_constraints = getattr(m, "_recovery_admissibility", None)
    if recovery_constraints:
        parts.append(
            "recovery_admissibility=%s"
            % ";".join(
                "%s=%r" % (name, recovery_constraints[name])
                for name in m.prim_state
                if name in recovery_constraints
            )
        )
    flux_axes = _axis_names(m._flux, where="model_hash flux")
    eig_axes = _axis_names(m._eig, where="model_hash eigenvalues") if m._eig else ()
    if eig_axes and eig_axes != flux_axes:
        raise ValueError("model_hash eigenvalue axes differ from physical flux axes")
    for d in flux_axes:
        parts.append("flux_%s=%s" % (d, ";".join(repr(e) for e in m._flux.get(d, []))))
        parts.append("eig_%s=%s" % (d, ";".join(repr(e) for e in m._eig.get(d, []))))
    parts.append("source=%s" % (";".join(repr(e) for e in m._source) if m._source else ""))
    if getattr(m, "_source_terms", None):
        parts.append(
            "source_terms=%s"
            % ";".join(
                "%s:[%s]" % (k, ",".join(repr(e) for e in m._source_terms[k]))
                for k in sorted(m._source_terms)
            )
        )
    if getattr(m, "_linear_sources", None):
        parts.append(
            "linear_sources=%s"
            % ";".join(
                "%s:[%s]" % (k, ";".join(repr(e) for row in m._linear_sources[k] for e in row))
                for k in sorted(m._linear_sources)
            )
        )
    if getattr(m, "_local_transforms", None):
        from pops._ir.visitors import _dag_key_data

        parts.append(
            "local_transforms=%s"
            % ";".join(
                "%s:%r"
                % (
                    k,
                    _dag_key_data(
                        (*m._local_transforms[k]["expressions"], m._local_transforms[k]["valid_if"])
                    ),
                )
                for k in sorted(m._local_transforms)
            )
        )
    if getattr(m, "_flux_terms", None):
        parts.append(
            "flux_terms=%s"
            % ";".join(
                "%s:%s"
                % (
                    k,
                    ":".join(
                        "%s[%s]"
                        % (
                            axis,
                            ",".join(repr(e) for e in m._flux_terms[k][axis]),
                        )
                        for axis in _axis_names(
                            m._flux_terms[k], where="model_hash named flux %r" % k
                        )
                    ),
                )
                for k in sorted(m._flux_terms)
            )
        )
    parts.append("cons_from=%s" % (";".join(repr(e) for e in m.cons_from) if m.cons_from else ""))
    parts.append("elliptic=%s" % (repr(m._elliptic) if m._elliptic is not None else ""))
    if getattr(m, "_elliptic_fields", None):
        parts.append(
            "elliptic_fields=%s"
            % ";".join(
                "%s:%s:%s:[%s]:gradient_sign=%d"
                % (
                    k,
                    m._elliptic_fields[k]["operator"],
                    repr(m._elliptic_fields[k]["rhs"]),
                    ",".join(m._elliptic_fields[k]["aux"]),
                    m._elliptic_fields[k]["gradient_sign"],
                )
                for k in sorted(m._elliptic_fields)
            )
        )
    parts.append("stab_speed=%s" % (repr(m._stab_speed) if m._stab_speed is not None else ""))
    parts.append("stab_dt=%s" % (repr(m._stab_dt) if m._stab_dt is not None else ""))
    parts.append("src_freq=%s" % (repr(m._src_freq) if m._src_freq is not None else ""))
    parts.append(
        "src_jac=%s"
        % (";".join(repr(e) for row in m._src_jac for e in row) if m._src_jac is not None else "")
    )
    if getattr(m, "_proj", None) is not None:
        parts.append("proj=%s" % ";".join(repr(e) for e in m._proj))
    from pops.numerics.riemann.providers import authoring_provider_evidence

    riemann_evidence = authoring_provider_evidence(m)
    parts.append("hllc=%d" % (1 if m._hllc else 0))
    if riemann_evidence.hllc_provider is not None:
        parts.append("hllc_provider=%s" % riemann_evidence.hllc_provider)
    forms = getattr(m, "_riemann_hook_forms", None)
    if forms:
        parts.append("riemann_hooks=%s" % ";".join("%s=%r" % (k, forms[k]) for k in sorted(forms)))
    parts.append("roe=%d" % (1 if getattr(m, "_roe", False) else 0))
    if riemann_evidence.roe_provider is not None:
        parts.append("roe_provider=%s" % riemann_evidence.roe_provider)
        parts.append("roe_entropy_policy=%s" % riemann_evidence.roe_entropy_policy)
        if riemann_evidence.roe_entropy_delta is not None:
            parts.append("roe_entropy_delta=%s" % riemann_evidence.roe_entropy_delta)
    if getattr(m, "_roe_rows", None) is not None:
        roe_axes = _axis_names(m._roe_rows, where="model_hash Roe rows")
        parts.append("roe_rows=%s" % ";".join(repr(e) for k in roe_axes for e in m._roe_rows[k]))
    if getattr(m, "_roe_jacobian", None) is not None:
        from pops.codegen.module_emit_riemann import has_characteristic_no_inflow_provider

        if has_characteristic_no_inflow_provider(m):
            parts.append("characteristic_no_inflow=flux_jacobian_v1")
        roe_jac_axes = _axis_names(
            {key: value for key, value in m._roe_jacobian.items() if key in ("x", "y", "z")},
            where="model_hash Roe Jacobian",
        )
        parts.append(
            "roe_jac=%s"
            % ";".join(repr(e) for k in roe_jac_axes for row in m._roe_jacobian[k] for e in row)
        )
        entropy_fix = m._roe_jacobian.get("entropy_fix")
        if entropy_fix is not None:
            parts.append("roe_jac_entropy_fix=%s" % _scalar_token(entropy_fix))
    if getattr(m, "_wave_speeds", None) is not None:
        wave_axes = _axis_names(m._wave_speeds, where="model_hash wave speeds")
        parts.append(
            "wave_speeds=%s" % ";".join(repr(e) for k in wave_axes for e in m._wave_speeds[k])
        )
    if getattr(m, "_ws_jacobian", None) is not None:
        # Model validation guarantees the closed Jacobian carrier shape.  Keep that runtime
        # authority intact while making the mapping contract explicit to the type checker.
        ws = cast(Mapping[str, Any], m._ws_jacobian)
        ws_axes = _axis_names(ws["blocks"], where="model_hash wave-speed blocks")
        parts.append(
            "ws_jac=%s|%s|%s"
            % (
                ws["eig"],
                "//".join(
                    ";".join(",".join(str(i) for i in b) for b in ws["blocks"][k]) for k in ws_axes
                ),
                ";".join(repr(e) for k in ws_axes for row in ws["rows"][k] for e in row)
                if ws["rows"] is not None
                else "",
            )
        )
        # ADC-617: fd_eps is EMITTED into the eig='fd' Jacobian, so it MUST enter the model hash or two
        # models differing only in fd_eps would collide on the same cached .so and serve wrong numerics.
        # The central-per-column-v4 marker invalidates artifacts emitted before each column used
        # its own scale, a unit floor and a symmetric perturbation (the former route used U[0], no
        # usable zero-state scale and forward FD).
        if ws["eig"] == "fd":
            parts.append("ws_jac_fd_step=central_per_column_v4")
        if ws.get("fd_eps") is not None:
            parts.append("ws_jac_fd_eps=%s" % _scalar_token(ws["fd_eps"]))
        # ADC-645: eig_max_iter / im_tol are EMITTED into the eig kernels (real_eig_minmax /
        # roe_abs_apply args), so they enter the hash -- but ONLY when set, keeping the default
        # model_hash byte-identical (the fd_eps rule).
        if ws.get("eig_max_iter") is not None:
            parts.append("ws_jac_eig_max_iter=%d" % int(ws["eig_max_iter"]))
        if ws.get("im_tol") is not None:
            parts.append("ws_jac_im_tol=%s" % _scalar_token(ws["im_tol"]))
    provider_metadata = getattr(m, "_auxiliary_provider_metadata", None)
    if provider_metadata is None:
        raise ValueError(
            "model_hash requires the exact auxiliary ProviderPack; compile through Module"
        )
    parts.append(
        "aux_provider_pack=%s"
        % json.dumps(provider_metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    )
    parts.append("gamma=%s" % ("None" if m.gamma is None else _scalar_token(m.gamma)))
    params = params or {}
    param_rows = []
    for key in sorted(params):
        declaration = params[key]
        artifact_data = getattr(declaration, "artifact_data", None)
        if not callable(artifact_data):
            raise TypeError(
                "model parameters must be canonical RuntimeParam/ConstParam/DerivedParam "
                "declarations; %r has no artifact_data()" % type(declaration).__name__
            )
        row = artifact_data()
        if not isinstance(row, Mapping):
            raise TypeError(
                "model parameter artifact_data() must return a mapping; got %s" % type(row).__name__
            )
        if row.get("name") != key:
            raise ValueError(
                "parameter registry key %r does not match declaration name %r"
                % (key, row.get("name"))
            )
        param_rows.append(row)
    parts.append(
        "params=%s"
        % json.dumps(param_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# _emit_route_manifest -- embedded native route registry signature (ADC-599)
# ---------------------------------------------------------------------------


def _emit_route_manifest(symbol_name: Any) -> str:
    """Emit the mandatory native route-registry signature (ADC-599).

    Returns the C++ source of ``extern "C" const char* <symbol_name>()`` returning
    ``route_registry_signature()`` evaluated at EMIT time (the versioned semantic-catalog digest,
    byte-identical to ``pops::route_registry_signature()``). The C++ loader requires this symbol and calls
    pops::verify_route_manifest(value, ctx): a stale .so built against a different route set, or an
    artifact predating this contract, is refused before any run. Imported here (routes.py is
    behavior-only) so the signature is baked into the string, exactly like pops_native_abi_key bakes
    POPS_ABI_KEY_LITERAL.
    """
    from pops.runtime.routes import route_registry_signature

    return 'extern "C" const char* %s() { return "%s"; }\n' % (
        symbol_name,
        route_registry_signature(),
    )


def _consumer_owner_qid(model: Any, consumer_owner_qid: Any = None) -> str:
    """Return the official consumer-plan owner; providers stay model-owned."""
    if consumer_owner_qid is None or consumer_owner_qid == "":
        owner = getattr(model, "owner_path", None)
        canonical = getattr(owner, "canonical", None)
        if not callable(canonical):
            raise ValueError("auxiliary consumer emission requires a canonical owner path")
        return str(canonical())
    if not isinstance(consumer_owner_qid, str) or not consumer_owner_qid:
        raise ValueError("auxiliary consumer owner qid must be a non-empty official owner path")
    return consumer_owner_qid


def _emit_auxiliary_route_registration(
    model: Any,
    *,
    target: str = "system",
    consumer_owner_qid: Any = None,
    declare_auxiliary_providers: bool = True,
) -> str:
    """Emit one DSO hook that registers, but never seals, auxiliary routes.

    The host stages this canonical hook with *every* package, invokes all
    registrars inside the native finalization transaction, then seals the one
    global registry and only afterwards installs prepared blocks.  That ordering
    is essential for dependencies spanning blocks and exact DSO-launcher rollback.
    """
    if target not in ("system", "amr_system"):
        raise ValueError("auxiliary route emission target must be 'system' or 'amr_system'")
    pack = getattr(model, "_auxiliary_provider_metadata", None)
    plans = getattr(model, "_component_operator_consumer_plans", None)
    flux_plan = getattr(model, "_component_flux_consumer_plan", None)
    if pack is None or plans is None or flux_plan is None:
        raise ValueError("native auxiliary route emission requires resolved ProviderPack metadata")
    if not isinstance(pack, Mapping) or not isinstance(pack.get("entries"), list):
        raise TypeError("auxiliary ProviderPack metadata has an invalid schema")
    routes = getattr(model, "_auxiliary_provider_routes", None)
    if routes is None:
        raise ValueError("native auxiliary route emission requires resolved typed producer routes")

    def route_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
        value = row["key"]
        return tuple(
            value[name]
            for name in (
                "owner_qid",
                "space_kind",
                "space_name",
                "component",
            )
        )

    typed_routes = {
        (key.owner_qid, key.space_kind, key.space_name, key.component): value
        for key, value in routes.items()
    }

    def literal(value: Any) -> str:
        return json.dumps(value)

    def optional(value: Any) -> str:
        if value is None:
            return "std::nullopt"
        return "std::optional<std::string>{%s}" % literal(value)

    def key(row: Mapping[str, Any]) -> str:
        value = row["key"]
        return "Key{%s, %s, %s, %s}" % tuple(
            literal(value[name]) for name in ("owner_qid", "space_kind", "space_name", "component")
        )

    def contract(row: Mapping[str, Any]) -> str:
        value = row["contract"]
        return "Contract{%s, %s, %s, %s, %s}" % (
            literal(value["representation"]),
            literal(value["centering"]),
            optional(value["unit"]),
            literal(value["layout"]),
            optional(value["value_kind"]),
        )

    def shape_for(route: Mapping[str, Any] | None) -> str:
        boundary = None if route is None else route.get("boundary")
        width = 0 if boundary is None else boundary.width
        return (
            "Shape{pops::kNativeDimension, 1, [] { pops::Index<pops::kNativeDimension> halo{}; "
            "for (int axis = 0; axis < pops::kNativeDimension; ++axis) halo[axis] = %d; return halo; }()}"
            % width
        )

    def boundary_for(route: Mapping[str, Any] | None) -> str:
        from pops.identity.scalar import scalar_cpp

        boundary = None if route is None else route.get("boundary")
        if boundary is None:
            return "Boundary{BoundaryKind::inherit_topology, std::nullopt}"
        kinds = {
            "inherit": "BoundaryKind::inherit_topology",
            "foextrap": "BoundaryKind::first_order_extrapolation",
            "dirichlet": "BoundaryKind::dirichlet",
        }
        try:
            kind = kinds[boundary.kind]
        except KeyError:
            raise ValueError("unsupported AuxiliaryBoundary kind %r" % boundary.kind) from None
        value = (
            "std::optional<pops::Real>{%s}" % scalar_cpp(boundary.value)
            if boundary.kind == "dirichlet"
            else "std::nullopt"
        )
        return "Boundary{%s, %s}" % (kind, value)

    def derived_expression_cpp(expression: Any, bindings: Mapping[str, str]) -> str:
        """Lower the compact scalar Expr subset accepted by native aux kernels.

        ``ValueExpr`` has intentionally no context-free C++ spelling.  Here it
        is bound only through the exact dependency vector of the derived route;
        free-name ``Var`` and state/parameter reads are rejected rather than
        becoming a hidden carrier lookup.
        """
        from pops._ir.expr import Abs, Const, Div, Maximum, Minimum, Mul, Neg, Pow, Sqrt, Sub, Add
        from pops._ir.handle_expr import ValueExpr

        if isinstance(expression, Const):
            return expression.to_cpp()
        if isinstance(expression, ValueExpr):
            try:
                return bindings[expression.handle.qualified_id]
            except KeyError:
                raise ValueError(
                    "DerivedAux expression reads undeclared dependency %s"
                    % expression.handle.qualified_id
                ) from None
        binary = (Add, Sub, Mul, Div)
        if isinstance(expression, binary):
            return "(%s %s %s)" % (
                derived_expression_cpp(expression.a, bindings),
                expression.op,
                derived_expression_cpp(expression.b, bindings),
            )
        if isinstance(expression, Pow):
            return "Kokkos::pow(%s, %s)" % (
                derived_expression_cpp(expression.a, bindings),
                derived_expression_cpp(expression.b, bindings),
            )
        if isinstance(expression, Minimum):
            return "Kokkos::fmin(%s, %s)" % (
                derived_expression_cpp(expression.a, bindings),
                derived_expression_cpp(expression.b, bindings),
            )
        if isinstance(expression, Maximum):
            return "Kokkos::fmax(%s, %s)" % (
                derived_expression_cpp(expression.a, bindings),
                derived_expression_cpp(expression.b, bindings),
            )
        if isinstance(expression, Neg):
            return "(-%s)" % derived_expression_cpp(expression.a, bindings)
        if isinstance(expression, Sqrt):
            return "Kokkos::sqrt(%s)" % derived_expression_cpp(expression.a, bindings)
        if isinstance(expression, Abs):
            return "Kokkos::abs(%s)" % derived_expression_cpp(expression.a, bindings)
        raise TypeError(
            "DerivedAux native lowering supports scalar constants, ValueExpr, and standard "
            "pointwise arithmetic; got %s" % type(expression).__name__
        )

    def derived_launcher(identity: str, route: Mapping[str, Any]) -> str:
        dependencies = route["dependencies"]
        producer = route["producer"]
        bindings = {
            reference.qualified_id: "aux_dependency_%d" % index
            for index, reference in enumerate(producer.expression.declaration_references())
        }
        expression = derived_expression_cpp(producer.expression, bindings)
        dependency_rows = []
        for key_value, contract_value in zip(dependencies, route["contracts"], strict=True):
            dependency_rows.append(
                "Dependency{%s, Contract{%s, %s, %s, %s, %s}, %s}"
                % (
                    "Key{%s, %s, %s, %s}"
                    % tuple(
                        literal(value)
                        for value in (
                            key_value.owner_qid,
                            key_value.space_kind,
                            key_value.space_name,
                            key_value.component,
                        )
                    ),
                    literal(contract_value.representation),
                    literal(contract_value.centering),
                    optional(contract_value.unit),
                    literal(contract_value.layout),
                    optional(contract_value.value_kind),
                    shape_for(
                        typed_routes.get(
                            (
                                key_value.owner_qid,
                                key_value.space_kind,
                                key_value.space_name,
                                key_value.component,
                            )
                        )
                    ),
                )
            )
        lines = [
            "      std::vector<Dependency>{%s}," % ", ".join(dependency_rows),
            "      Provider::launcher_type::trusted_extension(",
            "          pops::PreparedProviderIdentity{%s, 1}, %s,"
            % (
                literal("pops.derived-aux." + identity),
                literal(identity),
            ),
            "          [](const pops::runtime::system::AuxiliaryKernelLaunchContext<"
            "pops::kNativeDimension>& context) {",
            '            if (context.outputs.size() != 1) throw std::logic_error("derived auxiliary route requires one output");',
            '            if (context.dependencies.size() != %d) throw std::logic_error("derived auxiliary route dependency mismatch");'
            % len(dependencies),
            "            auto* const candidate = context.storage.candidate;",
            '            if (candidate == nullptr) throw std::logic_error("derived auxiliary route has no candidate storage groups");',
            "            auto* const output_group = candidate->find(context.outputs[0].address.group);",
            '            if (output_group == nullptr) throw std::logic_error("derived auxiliary output group is absent");',
            "            const auto output_component = context.outputs[0].address.component;",
        ]
        for index in range(len(dependencies)):
            lines.extend(
                (
                    "            const auto* const dependency_group_%d = candidate->find(context.dependencies[%d].address.group);"
                    % (index, index),
                    '            if (dependency_group_%d == nullptr || dependency_group_%d->layout() != output_group->layout() || dependency_group_%d->distribution() != output_group->distribution() || dependency_group_%d->local_rank() != output_group->local_rank()) throw std::logic_error("derived auxiliary dependency storage is not cell-compatible with its output");'
                    % (index, index, index, index),
                    "            const auto dependency_component_%d = context.dependencies[%d].address.component;"
                    % (index, index),
                )
            )
        lines.extend(
            (
                "            for (std::size_t local_fab = 0; local_fab < output_group->local_size(); ++local_fab) {",
                "              const auto output = output_group->fab(local_fab).view();",
                "              std::size_t cells = 1;",
                "              for (int axis = 0; axis < pops::kNativeDimension; ++axis) cells *= static_cast<std::size_t>(output.extents[axis]);",
                '              Kokkos::parallel_for("pops_derived_aux", Kokkos::RangePolicy<>(0, cells), KOKKOS_LAMBDA(const std::size_t linear) {',
                "                std::size_t remainder = linear;",
                "                pops::Index<pops::kNativeDimension> index{};",
                "                for (int axis = 0; axis < pops::kNativeDimension; ++axis) {",
                "                  index[axis] = output.origin[axis] + static_cast<int>(remainder % static_cast<std::size_t>(output.extents[axis]));",
                "                  remainder /= static_cast<std::size_t>(output.extents[axis]);",
                "                }",
            )
        )
        for index in range(len(dependencies)):
            lines.append(
                "                const auto dependency_%d = dependency_group_%d->fab(local_fab).view();"
                % (index, index)
            )
        for index in range(len(dependencies)):
            lines.append(
                "                const pops::Real aux_dependency_%d = dependency_%d(index, dependency_component_%d);"
                % (index, index, index)
            )
        lines.extend(
            (
                "                output(index, output_component) = %s;" % expression,
                "              });",
                "            }",
                "          }))",
            )
        )
        return "\n".join(lines)

    native_type = (
        "pops::runtime::system::PreparedNativeRouteRegistrar<pops::kNativeDimension>"
        if target == "system"
        else "pops::AmrSystem<pops::kNativeDimension>"
    )
    hook = (
        "pops_register_provider_routes"
        if target == "system"
        else "pops_register_provider_routes_amr"
    )
    receiver = "void* receiver" if target == "system" else "%s* sys" % native_type
    lines = [
        "POPS_LOADER_API void %s(%s) {" % (hook, receiver),
        '  if (%s == nullptr) throw std::invalid_argument("auxiliary route installer received null exact runtime");'
        % ("receiver" if target == "system" else "sys"),
        "  auto* sys = static_cast<%s*>(receiver);" % native_type if target == "system" else "",
        "  using namespace pops::runtime::system;",
        "  using Key = AuxiliaryComponentKey;",
        "  using Contract = AuxiliaryComponentContract;",
        "  using Shape = AuxiliaryStorageShape<pops::kNativeDimension>;",
        "  using Boundary = AuxiliaryBoundaryPolicy;",
        "  using BoundaryKind = AuxiliaryBoundaryPolicy::Kind;",
        "  using Output = AuxiliaryOutput<pops::kNativeDimension>;",
        "  using Dependency = AuxiliaryDependency<pops::kNativeDimension>;",
        "  using Provider = PreparedAuxiliaryProvider<pops::kNativeDimension>;",
        "  using ConsumerValue = AuxiliaryConsumerValue<pops::kNativeDimension>;",
        "  using ConsumerPlan = AuxiliaryConsumerProviderPlan<pops::kNativeDimension>;",
    ]
    owned_provider_qid = str(model.owner_path.canonical())
    for row in pack["entries"] if declare_auxiliary_providers else ():
        if row["key"]["owner_qid"] != owned_provider_qid:
            # A package may consume a foreign owner-qualified component.  Its
            # consumer plan below records that dependency, but only the owning
            # DSO may publish the provider output into the global registry.
            continue
        value = row["provider"]
        if value["slot"] is None or not value["availability"] or value["producer"] is None:
            raise ValueError(
                "auxiliary ProviderPack has an unavailable component; bind an exact producer before codegen"
            )
        producer = value["producer"]
        route = typed_routes.get(route_key(row))
        shape = shape_for(route)
        if producer == "runtime_input":
            kind = "AuxiliaryProviderKind::input"
        elif row["key"]["space_kind"] == "field":
            # FieldSpace is the sole authority for native field outputs.  The
            # producer spelling is an opaque canonical operator identity (not
            # a path syntax), therefore classify by the typed space rather
            # than guessing from a slash or a historical field name.
            kind = "AuxiliaryProviderKind::field_output"
        elif producer == "derived" or producer.startswith("derived:"):
            if route is None or route.get("kind") != "derived":
                raise ValueError(
                    "derived auxiliary provider %r has no exact typed lowering route"
                    % row["key"]["component"]
                )
            kind = "AuxiliaryProviderKind::derived"
        else:
            raise ValueError(
                "auxiliary provider %r has unsupported producer %r; "
                "expected runtime_input, a FieldOperator/field-solve output, or a "
                "lowered derived launcher" % (row["key"]["component"], producer)
            )
        identity = "provider:%s:%s/%s/%s" % (
            value["producer"],
            row["key"]["owner_qid"],
            row["key"]["space_name"],
            row["key"]["component"],
        )
        policy = (
            "AuxiliaryEvaluationPolicy{AuxiliaryEvaluationEvent::before_residual, "
            "AuxiliaryFreshness::evaluation}"
            if kind == "AuxiliaryProviderKind::derived"
            else "AuxiliaryEvaluationPolicy{AuxiliaryEvaluationEvent::initialization, "
            "AuxiliaryFreshness::once}"
        )
        lines.extend(
            (
                "  sys->install_prepared_auxiliary_provider(Provider{",
                "      %s, %s," % (literal(identity), kind),
                "      %s," % policy,
                "      std::vector<Output>{{%s, %s, %s, %s}},"
                % (key(row), contract(row), shape, boundary_for(route)),
            )
        )
        if kind == "AuxiliaryProviderKind::derived":
            if route is None:
                raise ValueError(
                    "derived auxiliary provider %r has no exact typed lowering route"
                    % row["key"]["component"]
                )
            lines.append(derived_launcher(identity, route) + ");")
        else:
            lines.append("      std::vector<Dependency>{}});")

    owner_qid = _consumer_owner_qid(model, consumer_owner_qid)

    def emit_plan(identity: str, plan: Any) -> None:
        lines.append(
            "  sys->install_auxiliary_consumer_plan(ConsumerPlan{%s, std::vector<ConsumerValue>{"
            % literal(identity)
        )
        for value in plan:
            lines.append(
                "      ConsumerValue{Dependency{%s, %s, %s}, %d},"
                % (
                    key(value),
                    contract(value),
                    shape_for(typed_routes.get(route_key(value))),
                    value["consumer_slot"],
                )
            )
        lines.append("  }});")

    emit_plan(owner_qid + "/physical_flux", flux_plan)
    for operator, plan in plans.items():
        emit_plan(owner_qid + "/operator/" + operator, plan)
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Native source emitter
# ---------------------------------------------------------------------------


def emit_cpp_native_loader(
    model: Any,
    name: Any = None,
    target: Any = "system",
    hoist_reciprocals: Any = False,
    model_identity: Any = None,
    native_field_roles: Any = None,
    consumer_owner_qid: Any = None,
    declare_auxiliary_providers: bool = True,
) -> str:
    """Source of the sole production package.

    The generated module carries the model and installs it directly into the native
    facade.  The complete, already-resolved BindSchema vector crosses the fixed ABI
    once and is injected before any closure is constructed.

    @p target: "system" (default) | "amr_system". Selects the targeted facade and
    thus the add_compiled_model OVERLOAD called.
    """
    from pops.codegen.cpp_writer import _cpp_identifier
    from pops.codegen.module_codegen import (
        _emit_bricks,
        _emit_metadata,
        _elliptic_field_registrations,
    )

    m = model
    if target not in ("system", "amr_system"):
        raise ValueError(
            "emit_cpp_native_loader: target 'system' | 'amr_system' (got %r)" % (target,)
        )
    if target == "system" and native_field_roles is not None:
        raise ValueError("resolved AMR field roles cannot be emitted into a System package")
    nv, bricks, composite = _emit_bricks(m, name, hoist_reciprocals=hoist_reciprocals)
    model_identity = str(model_identity if model_identity is not None else model_hash(m))
    if len(model_identity) != 64 or any(ch not in "0123456789abcdef" for ch in model_identity):
        raise ValueError("emit_cpp_native_loader requires one lowercase 64-hex model identity")
    nm = _cpp_identifier(name or (m.name.capitalize() + "Gen"))
    ell_field_regs = _elliptic_field_registrations(m, nm)
    amr_field_roles = _normalize_native_amr_field_roles(native_field_roles)
    if target == "amr_system" and ell_field_regs and native_field_roles is None:
        raise ValueError(
            "named AMR elliptic providers require exact resolved per-block field roles"
        )
    head = (
        "#include <Kokkos_Core.hpp>\n"
        "#include <cmath>\n"
        "#include <vector>\n"
        "#include <array>\n"
        "#include <cstddef>\n"
        "#include <optional>\n"
        "#include <stdexcept>\n"
        "#include <string>\n"
        "#include <utility>\n"
        "#include <pops/runtime/dynamic/abi_key.hpp>\n"
        "#include <pops/core/foundation/native_dimension.hpp>\n"
        "#include <pops/runtime/builders/compiled/model_runtime_params.hpp>\n"
        "#include <pops/physics/bricks/bricks.hpp>\n"
        "#include <pops/core/state/variables.hpp>\n"
    )
    head += (
        "#include <pops/runtime/builders/compiled/dsl_block.hpp>\n"
        if target == "system"
        else "#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>\n"
    )
    key = (
        "#if defined(_WIN32)\n"
        '#define POPS_LOADER_API extern "C" __declspec(dllexport)\n'
        "#else\n"
        '#define POPS_LOADER_API extern "C"\n'
        "#endif\n"
        "POPS_LOADER_API const char* pops_native_abi_key() {\n"
        "  return POPS_ABI_KEY_LITERAL;\n"
        "}\n"
        "POPS_LOADER_API const char* pops_compiled_model_identity() {\n"
        '  return "%s";\n'
        "}\n" % model_identity
    )
    if target == "system":
        key += (
            "POPS_LOADER_API int pops_native_system_package_abi_version() {\n"
            "  return pops::runtime::system::kNativeSystemPackageAbiVersion;\n"
            "}\n"
        )
    # Construct every elliptic RHS closure while ``model`` still owns the runtime parameters bound
    # from BindSchema.  The default field must capture the generated CompositeModel: its Ell brick
    # intentionally exposes only rhs(State), while CompositeModel supplies State + elliptic_rhs.
    # Named-field bricks are standalone models, so copy the same bound RuntimeParams carrier into
    # each one before type-erasing it. Every closure is then moved into the inert complete package;
    # the host validates its block and elliptic attachments before one atomic commit.
    system_elliptic_prepare_lines = ""
    system_elliptic_package_lines = ""
    for index, (fld, brick, output_keys) in enumerate(ell_field_regs):
        gradient_sign = m._elliptic_fields[fld]["gradient_sign"]
        if type(gradient_sign) is not int or gradient_sign not in (-1, 1):
            raise ValueError("elliptic_field('%s'): gradient_sign must be exactly -1 or 1" % fld)
        if len(output_keys) == 1 and gradient_sign != 1:
            raise ValueError(
                "elliptic_field('%s'): gradient_sign=-1 requires gradient outputs" % fld
            )
        system_elliptic_prepare_lines += (
            "  auto named_elliptic_model_%d = %s{};\n"
            "  pops::compiled_model::apply_runtime_params(\n"
            "      named_elliptic_model_%d,\n"
            "      pops::compiled_model::declaration_runtime_params(model));\n"
            "  auto named_elliptic_rhs_%d = pops::make_poisson_rhs(named_elliptic_model_%d);\n"
            % (index, brick, index, index, index)
        )
        key_values = ", ".join(
            "pops::runtime::system::AuxiliaryComponentKey{%s, %s, %s, %s}"
            % tuple(
                json.dumps(value)
                for value in (
                    key.owner_qid,
                    key.space_kind,
                    key.space_name,
                    key.component,
                )
            )
            for key in output_keys
        )
        system_elliptic_package_lines += (
            '  package.elliptic_attachments.push_back({"%s", "%s/%s", '
            "std::vector<pops::runtime::system::AuxiliaryComponentKey>{%s}, %d, "
            "std::move(named_elliptic_rhs_%d)});\n"
            % (fld, model_identity, fld, key_values, gradient_sign, index)
        )
    if m._elliptic is not None:
        system_elliptic_prepare_lines += (
            "  auto fields_from_state_rhs = pops::make_poisson_rhs(model);\n"
        )
        system_elliptic_package_lines += (
            '  package.elliptic_attachments.push_back({"fields_from_state", '
            '"%s/fields_from_state", {}, 1, std::move(fields_from_state_rhs)});\n' % model_identity
        )

    amr_elliptic_prepare_lines = ""
    amr_elliptic_package_lines = ""
    local_fields = {field: brick for field, brick, _ in ell_field_regs}
    rhs_index = 0
    for role in amr_field_roles:
        if role["kind"] == "output":
            key_values = ", ".join(
                "pops::runtime::system::AuxiliaryComponentKey{%s, %s, %s, %s}"
                % tuple(
                    json.dumps(key[name])
                    for name in ("owner_qid", "space_kind", "space_name", "component")
                )
                for key in role["output_keys"]
            )
            amr_elliptic_package_lines += (
                "  {\n"
                "    pops::PreparedNativeAmrEllipticAttachment<pops::kNativeDimension> attachment;\n"
                "    attachment.field = %s;\n"
                "    attachment.block_identity = %s;\n"
                "    attachment.output_keys = {%s};\n"
                "    attachment.gradient_sign = %d;\n"
                "    package.elliptic_attachments.push_back(std::move(attachment));\n"
                "  }\n"
                % (
                    json.dumps(role["field"]),
                    json.dumps(role["block"]),
                    key_values,
                    role["gradient_sign"],
                )
            )
            continue
        provider_key = role["provider_key"]
        brick = local_fields.get(provider_key)
        if brick is None:
            raise ValueError(
                "resolved AMR field RHS provider %r is absent from its block model" % provider_key
            )
        from pops.identity.scalar import scalar_cpp

        amr_elliptic_prepare_lines += (
            "  auto named_elliptic_model_%d = %s{};\n"
            "  pops::compiled_model::apply_runtime_params(\n"
            "      named_elliptic_model_%d,\n"
            "      pops::compiled_model::declaration_runtime_params(model));\n"
            "  auto named_elliptic_rhs_%d = pops::make_poisson_rhs(named_elliptic_model_%d);\n"
            % (rhs_index, brick, rhs_index, rhs_index, rhs_index)
        )
        amr_elliptic_package_lines += (
            "  {\n"
            "    pops::PreparedNativeAmrEllipticAttachment<pops::kNativeDimension> attachment;\n"
            "    attachment.field = %s;\n"
            "    attachment.block_identity = %s;\n"
            "    attachment.binding_identity = %s;\n"
            "    attachment.rhs_provider_identity = %s;\n"
            "    attachment.rhs_provider_key = %s;\n"
            "    attachment.binding_ordinal = %d;\n"
            "    attachment.coefficient = %s;\n"
            "    attachment.rhs = std::move(named_elliptic_rhs_%d);\n"
            "    package.elliptic_attachments.push_back(std::move(attachment));\n"
            "  }\n"
            % (
                json.dumps(role["field"]),
                json.dumps(role["block"]),
                json.dumps(role["binding_identity"]),
                json.dumps("%s/%s" % (model_identity, provider_key)),
                json.dumps(provider_key),
                role["binding_ordinal"],
                scalar_cpp(role["coefficient"]),
                rhs_index,
            )
        )
        rhs_index += 1
    if m._elliptic is not None:
        amr_elliptic_prepare_lines += (
            "  auto fields_from_state_rhs = pops::make_poisson_rhs(model);\n"
        )
        amr_elliptic_package_lines += (
            "  {\n"
            "    pops::PreparedNativeAmrEllipticAttachment<pops::kNativeDimension> attachment;\n"
            '    attachment.field = "fields_from_state";\n'
            '    attachment.rhs_provider_identity = "%s/fields_from_state";\n'
            "    attachment.coefficient = 1.0;\n"
            "    attachment.rhs = std::move(fields_from_state_rhs);\n"
            "    package.elliptic_attachments.push_back(std::move(attachment));\n"
            "  }\n" % model_identity
        )
    if target == "system":
        install = (
            "POPS_LOADER_API void pops_install_native(void* sys, const char* name, const char* limiter,\n"
            "                                    const char* riemann, const char* recon,\n"
            "                                    const char* time, double gamma, int substeps,\n"
            "                                    int evolve, int stride, const double* params,\n"
            "                                    int nparams, double pos_floor) {\n"
            "  using Installer = pops::runtime::system::PreparedNativeBlockInstaller<pops::kNativeDimension>;\n"
            "  auto* s = static_cast<Installer*>(sys);\n"
            "  auto model = pops::compiled_model::bind_runtime_params(\n"
            "      pops_generated::ProdModel{}, params, nparams);\n"
            + system_elliptic_prepare_lines
            + "  pops::runtime::system::PreparedNativeSystemPackage<pops::kNativeDimension> package;\n"
            "  package.consumer_qid = "
            + json.dumps(_consumer_owner_qid(m, consumer_owner_qid) + "/physical_flux")
            + ";\n"
            "  package.block = pops::prepare_compiled_system_block<pops::kNativeDimension>(*s, name, package.consumer_qid, std::move(model),\n"
            "      limiter, riemann, recon, time, gamma, substeps, evolve != 0, stride, pos_floor);\n"
            + system_elliptic_package_lines
            + "  s->commit(std::move(package));\n"
            "}\n"
        )
    else:
        # The untrusted installer may submit only one inert complete package. It cannot call
        # ordinary AmrSystem composition seams while the host lifecycle is Bound; the outer host
        # transaction witnesses the staged block and all elliptic attachments after callback return.
        install = (
            "POPS_LOADER_API void pops_install_native_amr(void* sys, const char* name,\n"
            "                                        const char* limiter, const char* riemann,\n"
            "                                        const char* recon, const char* time,\n"
            "                                        double gamma, int substeps,\n"
            "                                        const double* params, int nparams,\n"
            "                                        double pos_floor, double weno_epsilon,\n"
            "                                        bool wave_speed_cache) {\n"
            "  using NativeAmrSystem = pops::AmrSystem<pops::kNativeDimension>;\n"
            "  auto* s = reinterpret_cast<NativeAmrSystem*>(sys);\n"
            "  auto model = pops::compiled_model::bind_runtime_params(\n"
            "      pops_generated::ProdModel{}, params, nparams);\n"
            + amr_elliptic_prepare_lines
            + "  pops::PreparedNativeAmrPackage<pops::kNativeDimension> package;\n"
            "  package.block = pops::prepare_compiled_amr_system_block<pops::kNativeDimension>(\n"
            "      name, std::move(model), limiter, riemann, recon, time, gamma, substeps,\n"
            "      /*stride=*/1, pos_floor, weno_epsilon, wave_speed_cache, %s);\n"
            % json.dumps(_consumer_owner_qid(m, consumer_owner_qid) + "/physical_flux")
            + amr_elliptic_package_lines
            + "  s->install_prepared_native_amr_package(std::move(package));\n"
            "}\n"
        )
    install += (
        "POPS_LOADER_API int pops_compiled_nparams() {\n"
        "  return pops::compiled_model::runtime_param_count<pops_generated::ProdModel>();\n"
        "}\n"
        'POPS_LOADER_API const char* pops_compiled_param_names() { return "%s"; }\n'
        % ",".join(node.name for node in m.runtime_param_nodes())
    )
    package_preparer = ""
    if target == "system":
        package_preparer = (
            "\nnamespace pops_generated {\n"
            "static_assert(ProdModel::dimension == pops::kNativeDimension,\n"
            '              "generated model rank differs from the selected native artifact");\n'
            "inline pops::PreparedSystemBlock<pops::kNativeDimension> prepare_exact_system_block(\n"
            "    pops::CompiledSystemBlockPreparation<pops::kNativeDimension, ProdModel> request) {\n"
            "  return pops::prepare_generated_system_block(std::move(request));\n"
            "}\n"
            "}  // namespace pops_generated\n"
        )
    auxiliary_routes = _emit_auxiliary_route_registration(
        m,
        target=target,
        consumer_owner_qid=consumer_owner_qid,
        declare_auxiliary_providers=declare_auxiliary_providers,
    )
    return (
        head
        + bricks
        + "\nnamespace pops_generated { using ProdModel = %s; }\n" % composite
        + package_preparer
        + key
        + auxiliary_routes
        + install
        + _emit_metadata(m, "pops_generated::ProdModel")
        + _emit_route_manifest("pops_compiled_route_manifest")
    )
