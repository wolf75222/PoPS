"""C++ source emission for the sole native production package."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Backend / capability tables (single source of truth in this module)
# ---------------------------------------------------------------------------

_BACKEND_CAPS = {
    "production": {"cpu": True, "mpi": True, "amr": True, "gpu": False, "tier": "production"},
}


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
    from pops._ir.literals import scalar_data
    from pops._ir.values import _EIG_FIELDS  # noqa: F401 -- confirm ir is importable

    def _scalar_token(value: Any) -> str:
        return json.dumps(scalar_data(value), sort_keys=True, separators=(",", ":"))

    # --- lazy helpers: resolve at call time, not at import time ---
    def _aux_total_n_aux(aux_names: Any, aux_extra_names: Any) -> int:
        # Mirrors pops.dsl.aux_total_n_aux without importing dsl.
        _AUX_CANONICAL = {"phi": 0, "grad_x": 1, "grad_y": 2, "B_z": 3, "T_e": 4}
        _AUX_BASE_COMPS = 3
        _AUX_NAMED_BASE = 5
        w = _AUX_BASE_COMPS
        for nm in aux_names:
            if nm not in _AUX_CANONICAL:
                raise ValueError("unknown aux field %r" % (nm,))
            w = max(w, _AUX_CANONICAL[nm] + 1)
        if aux_extra_names:
            w = max(w, _AUX_NAMED_BASE + len(aux_extra_names))
        return w

    def _role_of(name: Any) -> str:
        _CANONICAL_ROLES = {
            "rho": "Density", "n": "Density", "density": "Density",
            "rho_u": "MomentumX", "rhou": "MomentumX", "mom_x": "MomentumX", "mx": "MomentumX",
            "rho_v": "MomentumY", "rhov": "MomentumY", "mom_y": "MomentumY", "my": "MomentumY",
            "rho_w": "MomentumZ", "rhow": "MomentumZ", "mom_z": "MomentumZ", "mz": "MomentumZ",
            "E": "Energy", "rho_E": "Energy", "ener": "Energy", "energy": "Energy",
            "u": "VelocityX", "v": "VelocityY", "w": "VelocityZ",
            "vx": "VelocityX", "vy": "VelocityY", "vz": "VelocityZ",
            "p": "Pressure", "pressure": "Pressure",
            "T": "Temperature", "temperature": "Temperature",
        }
        return _CANONICAL_ROLES.get(name, "Custom")

    def _roles_for(names: Any, override: Any = None) -> list:
        if override is None:
            return [_role_of(nm) for nm in names]
        if len(override) != len(names):
            raise ValueError("roles: %d roles for %d variables" % (len(override), len(names)))
        return [(r if r is not None else _role_of(nm)) for nm, r in zip(names, override, strict=True)]

    m = model
    parts = []
    parts.append("name=%s" % m.name)
    parts.append("cons=%s" % ",".join(m.cons_names))
    parts.append("croles=%s" % ",".join(_roles_for(m.cons_names, m.cons_roles)))
    parts.append("prim_state=%s" % ",".join(m.prim_state))
    parts.append("proles=%s" % ",".join(_roles_for(m.prim_state, m.prim_roles)))
    parts.append("prim=%s" % ";".join("%s=%r" % (k, m.prim_defs[k]) for k in m.prim_defs))
    for d in ("x", "y"):
        parts.append("flux_%s=%s" % (d, ";".join(repr(e) for e in m._flux.get(d, []))))
        parts.append("eig_%s=%s" % (d, ";".join(repr(e) for e in m._eig.get(d, []))))
    parts.append("source=%s" % (";".join(repr(e) for e in m._source) if m._source else ""))
    if getattr(m, "_source_terms", None):
        parts.append("source_terms=%s" % ";".join(
            "%s:[%s]" % (k, ",".join(repr(e) for e in m._source_terms[k]))
            for k in sorted(m._source_terms)))
    if getattr(m, "_linear_sources", None):
        parts.append("linear_sources=%s" % ";".join(
            "%s:[%s]" % (k, ";".join(repr(e) for row in m._linear_sources[k] for e in row))
            for k in sorted(m._linear_sources)))
    if getattr(m, "_flux_terms", None):
        parts.append("flux_terms=%s" % ";".join(
            "%s:x[%s]:y[%s]" % (k,
                                ",".join(repr(e) for e in m._flux_terms[k]["x"]),
                                ",".join(repr(e) for e in m._flux_terms[k]["y"]))
            for k in sorted(m._flux_terms)))
    parts.append("cons_from=%s" % (";".join(repr(e) for e in m.cons_from) if m.cons_from else ""))
    parts.append("elliptic=%s" % (repr(m._elliptic) if m._elliptic is not None else ""))
    if getattr(m, "_elliptic_fields", None):
        parts.append("elliptic_fields=%s" % ";".join(
            "%s:%s:%s:[%s]:gradient_sign=%d" % (
                k, m._elliptic_fields[k]["operator"],
                repr(m._elliptic_fields[k]["rhs"]),
                ",".join(m._elliptic_fields[k]["aux"]),
                m._elliptic_fields[k]["gradient_sign"])
            for k in sorted(m._elliptic_fields)))
    parts.append("stab_speed=%s" % (repr(m._stab_speed) if m._stab_speed is not None else ""))
    parts.append("stab_dt=%s" % (repr(m._stab_dt) if m._stab_dt is not None else ""))
    parts.append("src_freq=%s" % (repr(m._src_freq) if m._src_freq is not None else ""))
    parts.append("src_jac=%s" % (";".join(repr(e) for row in m._src_jac for e in row)
                                 if m._src_jac is not None else ""))
    if getattr(m, "_proj", None) is not None:
        parts.append("proj=%s" % ";".join(repr(e) for e in m._proj))
    parts.append("hllc=%d" % (1 if m._hllc else 0))
    forms = getattr(m, "_riemann_hook_forms", None)
    if forms:
        parts.append("riemann_hooks=%s" % ";".join(
            "%s=%r" % (k, forms[k]) for k in sorted(forms)))
    parts.append("roe=%d" % (1 if getattr(m, "_roe", False) else 0))
    if getattr(m, "_roe_rows", None) is not None:
        parts.append("roe_rows=%s" % ";".join(repr(e) for k in ("x", "y")
                                              for e in m._roe_rows[k]))
    if getattr(m, "_roe_jacobian", None) is not None:
        parts.append("roe_jac=%s" % ";".join(repr(e) for k in ("x", "y")
                                             for row in m._roe_jacobian[k] for e in row))
    if getattr(m, "_wave_speeds", None) is not None:
        parts.append("wave_speeds=%s" % ";".join(repr(e) for k in ("x", "y")
                                                 for e in m._wave_speeds[k]))
    if getattr(m, "_ws_jacobian", None) is not None:
        ws = m._ws_jacobian
        parts.append("ws_jac=%s|%s|%s" % (
            ws["eig"],
            "//".join(";".join(",".join(str(i) for i in b) for b in ws["blocks"][k])
                      for k in ("x", "y")),
            ";".join(repr(e) for k in ("x", "y") for row in ws["rows"][k] for e in row)
            if ws["rows"] is not None else ""))
        # ADC-617: fd_eps is EMITTED into the eig='fd' Jacobian, so it MUST enter the model hash or two
        # models differing only in fd_eps would collide on the same cached .so and serve wrong numerics.
        # Appended ONLY when set, so the default (None -> the historical 1e-6 literal) leaves the hash
        # byte-identical (no spurious cache miss for existing models).
        if ws.get("fd_eps") is not None:
            parts.append("ws_jac_fd_eps=%s" % _scalar_token(ws["fd_eps"]))
        # ADC-645: eig_max_iter / im_tol are EMITTED into the eig kernels (real_eig_minmax /
        # roe_abs_apply args), so they enter the hash -- but ONLY when set, keeping the default
        # model_hash byte-identical (the fd_eps rule).
        if ws.get("eig_max_iter") is not None:
            parts.append("ws_jac_eig_max_iter=%d" % int(ws["eig_max_iter"]))
        if ws.get("im_tol") is not None:
            parts.append("ws_jac_im_tol=%s" % _scalar_token(ws["im_tol"]))
    parts.append("n_aux=%d" % _aux_total_n_aux(m.aux_names, m.aux_extra_names))
    if m.aux_extra_names:
        parts.append("aux_extra=%s" % ",".join(m.aux_extra_names))
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
    return ('extern "C" const char* %s() { return "%s"; }\n'
            % (symbol_name, route_registry_signature()))


# ---------------------------------------------------------------------------
# Native source emitter
# ---------------------------------------------------------------------------


def emit_cpp_native_loader(model: Any, name: Any = None, target: Any = "system",
                           hoist_reciprocals: Any = False) -> str:
    """Source of the sole production package.

    The generated module carries the model and installs it directly into the native
    facade.  The complete, already-resolved BindSchema vector crosses the fixed ABI
    once and is injected before any closure is constructed.

    @p target: "system" (default) | "amr_system". Selects the targeted facade and
    thus the add_compiled_model OVERLOAD called.
    """
    from pops.codegen.module_codegen import _emit_bricks, _emit_metadata, _elliptic_field_registrations
    m = model
    if target not in ("system", "amr_system"):
        raise ValueError("emit_cpp_native_loader: target 'system' | 'amr_system' (got %r)"
                         % (target,))
    nv, bricks, composite = _emit_bricks(m, name, hoist_reciprocals=hoist_reciprocals)
    nm = name or (m.name.capitalize() + "Gen")
    ell_field_regs = _elliptic_field_registrations(m, nm)
    head = ('#include <cmath>\n'
            '#include <vector>\n'
            '#include <array>\n'
            '#include <cstddef>\n'
            '#include <string>\n'
            '#include <utility>\n'
            '#include <pops/runtime/dynamic/abi_key.hpp>\n'
            '#include <pops/runtime/builders/compiled/model_runtime_params.hpp>\n'
            '#include <pops/physics/bricks/bricks.hpp>\n'
            '#include <pops/core/state/variables.hpp>\n')
    head += ('#include <pops/runtime/builders/compiled/dsl_block.hpp>\n' if target == "system"
             else '#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>\n')
    key = ('#if defined(_WIN32)\n'
           '#define POPS_LOADER_API extern "C" __declspec(dllexport)\n'
           '#else\n'
           '#define POPS_LOADER_API extern "C"\n'
           '#endif\n'
           'POPS_LOADER_API const char* pops_native_abi_key() {\n'
           '  return POPS_ABI_KEY_LITERAL;\n'
           '}\n')
    # Construct every elliptic RHS closure while ``model`` still owns the runtime parameters bound
    # from BindSchema.  The default field must capture the generated CompositeModel: its Ell brick
    # intentionally exposes only rhs(State), while CompositeModel supplies State + elliptic_rhs.
    # Named-field bricks are standalone models, so copy the same bound RuntimeParams carrier into
    # each one before type-erasing it.  Attachment remains after add_compiled_model because the block
    # must exist before set_block_elliptic_field is called.
    ell_field_prepare_lines = ""
    ell_field_attach_lines = ""
    for index, (fld, brick, phi_c, gx_c, gy_c) in enumerate(ell_field_regs):
        gradient_sign = m._elliptic_fields[fld]["gradient_sign"]
        if type(gradient_sign) is not int or gradient_sign not in (-1, 1):
            raise ValueError(
                "elliptic_field('%s'): gradient_sign must be exactly -1 or 1" % fld)
        if gx_c < 0 and gradient_sign != 1:
            raise ValueError(
                "elliptic_field('%s'): gradient_sign=-1 requires gradient outputs" % fld)
        ell_field_prepare_lines += (
            '  auto named_elliptic_model_%d = %s{};\n'
            '  pops::compiled_model::apply_runtime_params(\n'
            '      named_elliptic_model_%d,\n'
            '      pops::compiled_model::declaration_runtime_params(model));\n'
            '  auto named_elliptic_rhs_%d = pops::make_poisson_rhs(named_elliptic_model_%d);\n'
            % (index, brick, index, index, index)
        )
        ell_field_attach_lines += (
            '  s->register_elliptic_field(name, "%s", %d, %d, %d, %d);\n'
            '  s->set_block_elliptic_field(name, "%s", std::move(named_elliptic_rhs_%d));\n'
            % (fld, phi_c, gx_c, gy_c, gradient_sign, fld, index)
        )
    if m._elliptic is not None:
        ell_field_prepare_lines += (
            '  auto fields_from_state_rhs = pops::make_poisson_rhs(model);\n'
        )
        ell_field_attach_lines += (
            '  s->set_block_elliptic_field(name, "fields_from_state", '
            'std::move(fields_from_state_rhs));\n'
        )
    if target == "system":
        install = ('POPS_LOADER_API void pops_install_native(void* sys, const char* name, const char* limiter,\n'
                   '                                    const char* riemann, const char* recon,\n'
                   '                                    const char* time, double gamma, int substeps,\n'
                   '                                    int evolve, int stride, const double* params,\n'
                   '                                    int nparams, double pos_floor) {\n'
                   '  pops::System* s = reinterpret_cast<pops::System*>(sys);\n'
                   '  auto model = pops::compiled_model::bind_runtime_params(\n'
                   '      pops_generated::ProdModel{}, params, nparams);\n'
                   + ell_field_prepare_lines +
                   '  pops::add_compiled_model<pops_generated::ProdModel>(*s, name, std::move(model),\n'
                   '                                                    limiter, riemann, recon, time, gamma,\n'
                   '                                                    substeps, evolve != 0, stride,\n'
                   '                                                    pos_floor);\n'
                   + ell_field_attach_lines +
                   '}\n')
    else:
        # NAMED elliptic fields on the AMR layout (ADC-428): mirror the uniform System branch, but on the
        # AmrSystem facade. register_elliptic_field records the field's aux output components and FORCES
        # the AmrRuntime engine (the named-field solve lives there); set_block_elliptic_field attaches the
        # per-field RHS brick to the block (name == this block). The default Poisson path is untouched.
        install = ('POPS_LOADER_API void pops_install_native_amr(void* sys, const char* name,\n'
                   '                                        const char* limiter, const char* riemann,\n'
                   '                                        const char* recon, const char* time,\n'
                   '                                        double gamma, int substeps,\n'
                   '                                        const double* params, int nparams,\n'
                   '                                        double pos_floor) {\n'
                   '  pops::AmrSystem* s = reinterpret_cast<pops::AmrSystem*>(sys);\n'
                   '  auto model = pops::compiled_model::bind_runtime_params(\n'
                   '      pops_generated::ProdModel{}, params, nparams);\n'
                   + ell_field_prepare_lines +
                   '  pops::add_compiled_model<pops_generated::ProdModel>(*s, name, std::move(model),\n'
                   '                                                    limiter, riemann, recon, time, gamma,\n'
                   '                                                    substeps, /*stride=*/1,\n'
                   '                                                    /*implicit_vars=*/{},\n'
                   '                                                    /*implicit_roles=*/{}, pos_floor);\n'
                   + ell_field_attach_lines +
                   '}\n')
    install += ('POPS_LOADER_API int pops_compiled_nparams() {\n'
                '  return pops::compiled_model::runtime_param_count<pops_generated::ProdModel>();\n'
                '}\n'
                'POPS_LOADER_API const char* pops_compiled_param_names() { return "%s"; }\n'
                % ",".join(node.name for node in m.runtime_param_nodes()))
    return (head
            + bricks
            + '\nnamespace pops_generated { using ProdModel = %s; }\n' % composite
            + key
            + install
            + _emit_metadata(m, "pops_generated::ProdModel")
            + _emit_route_manifest("pops_compiled_route_manifest"))
