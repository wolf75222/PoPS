"""pops.codegen.module_emit_helpers : shared codegen-only helpers for the module emitter.

Extracted verbatim from ``pops.codegen.module_codegen`` so the emit logic fits the
Spec-4 file-size budget; the emit functions in ``module_codegen`` and the Riemann
capability helpers in ``module_emit_riemann`` import these.  No circular import: this
module never imports pops.dsl or pops.physics at module level.

Contents
--------
_CANONICAL_ROLES, _role_of, _roles_for             -- role mirror (dsl.roles_for)
_ranked_axes, _axis_values                          -- exact Cartesian-rank helpers
_codegen_exprs, _live_prims, _prim_block, _jac_entries
"""
from __future__ import annotations

import json
from typing import Any

from pops._cartesian_axes import canonical_axis_mapping
from pops.codegen.cpp_writer import (
    _cse_emit,
    _count_cons_denoms,
    _recip_rewrite,
)
from pops._ir.visitors import _dependencies

# ---------------------------------------------------------------------------
# roles_for -- lazy delegation; avoids importing physics at module import time.
# ---------------------------------------------------------------------------
_CANONICAL_ROLES = {
    "rho": "density", "n": "density", "density": "density",
    "rho_u": "momentum:0", "rhou": "momentum:0", "mom_x": "momentum:0", "mx": "momentum:0",
    "rho_v": "momentum:1", "rhov": "momentum:1", "mom_y": "momentum:1", "my": "momentum:1",
    "rho_w": "momentum:2", "rhow": "momentum:2", "mom_z": "momentum:2", "mz": "momentum:2",
    "E": "energy", "rho_E": "energy", "ener": "energy", "energy": "energy",
    "u": "velocity:0", "v": "velocity:1", "w": "velocity:2",
    "vx": "velocity:0", "vy": "velocity:1", "vz": "velocity:2",
    "p": "pressure", "pressure": "pressure",
    "T": "temperature", "temperature": "temperature",
}


def _role_of(name: Any) -> str:
    return _CANONICAL_ROLES.get(name, "custom")


def _roles_for(names: Any, override: Any = None) -> list:
    """Lower typed authoring roles through the one structured-token authority."""
    from pops.physics.aux import roles_for

    return list(roles_for(names, override))


def _ranked_axes(model: Any) -> tuple[str, ...]:
    """Return the model's one canonical x[/y[/z]] rank authority."""
    return tuple(canonical_axis_mapping(model._flux, where="emit_cpp_brick flux").keys())


def _axis_values(model: Any, values: Any, *, where: str) -> list:
    """Flatten one exact-ranked carrier in the physical-flux axis order."""
    axes = _ranked_axes(model)
    if not isinstance(values, dict) or tuple(values) != axes:
        raise ValueError(
            "%s must cover the exact emitted axis set %s" % (where, axes)
        )
    return [item for axis in axes for item in values[axis]]


def _exact_brick_contract(
    model: Any,
    family: str,
    *,
    dimension: int,
    n_vars: int,
    runtime_params: bool,
    slot: str = "",
) -> list[str]:
    """Emit the host-side semantic contract owned by one generated physics brick."""
    model_hash = model._model_hash()
    if not isinstance(model_hash, str) or not model_hash:
        raise TypeError("generated physics bricks require a non-empty structural model hash")
    if not isinstance(family, str) or not family or not isinstance(slot, str):
        raise TypeError("generated physics brick family/slot identities must be strings")

    lines = [
        "  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() "
        "noexcept {",
        "    return {%s, 1};" % json.dumps("pops.codegen.%s-brick" % family),
        "  }",
        "  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {",
        "    contract.text(\"pops.codegen.exact-physics-brick\")",
        "        .scalar(std::uint32_t{1})",
        "        .text(%s)" % json.dumps(model_hash),
        "        .text(%s)" % json.dumps(slot),
        "        .scalar(std::int32_t{%d})" % dimension,
        "        .scalar(std::int32_t{%d});" % n_vars,
    ]
    if runtime_params:
        lines += [
            "    if (params.count < 0 || params.count > pops::kMaxRuntimeParams)",
            "      throw std::invalid_argument(\"generated physics brick runtime parameter count "
            "is invalid\");",
            "    contract.scalar(static_cast<std::int32_t>(params.count));",
            "    for (int index = 0; index < params.count; ++index)",
            "      contract.scalar(params.values[index]);",
        ]
    else:
        lines.append("    contract.scalar(std::int32_t{0});")
    lines += ["  }", ""]
    return lines


# ---------------------------------------------------------------------------
# Codegen-only helpers (used solely by the emit* functions)
# ---------------------------------------------------------------------------

def _codegen_exprs(model: Any, exprs: Any, cse: Any, real: str = "pops::Real", indent: str = "    ") -> tuple:
    """(CSE local lines, [C++ per expr]). If cse, factor the common subexpressions
    (H, c...) into ``cseK_`` locals ; otherwise inline each expression via to_cpp."""
    if cse:
        return _cse_emit(list(exprs), real, indent)
    return [], [e.to_cpp() for e in exprs]


def _live_prims(model: Any, exprs: Any, seed: Any = ()) -> set:
    """Names of the primitives transitively referenced by @p exprs (and the @p seed names).
    Closure over prim_defs: a live primitive pulls in its own primitive dependencies.
    Used to emit in a method only the primitives actually used (dead-code elimination):
    the live expressions stay identical, so the values are bit-identical."""
    prim = model.prim_defs
    live = set()
    stack = [n for n in seed if n in prim]
    stack.extend(name for name in _dependencies(exprs) if name in prim)
    while stack:
        nm = stack.pop()
        if nm in live:
            continue
        live.add(nm)
        stack.extend(name for name in _dependencies(prim[nm]) if name in prim)
    return live


def _prim_block(model: Any, live: Any = None, hoist: bool = False) -> list:
    """``const pops::Real <prim> = ...;`` lines of a method. @p live (default None = all):
    declares only the live primitives. @p hoist: hoists at the top the reciprocal of the
    recurring conservative denominators (>= 2 uses) and replaces those divisions by
    products (OPT-IN, changes the rounding). Without @p hoist and with live=None, historical output."""
    items = [(p, e) for p, e in model.prim_defs.items() if live is None or p in live]
    if not hoist:
        return ["    const pops::Real %s = %s;" % (p, e.to_cpp()) for p, e in items]
    cons_set = set(model.cons_names)
    counts = {}
    for _, e in items:
        _count_cons_denoms(e, cons_set, counts)
    inv = [n for n in model.cons_names if counts.get(n, 0) >= 2]  # stable cons order
    inv_set = set(inv)
    lines = ["    const pops::Real inv_%s = pops::Real(1) / %s;" % (n, n) for n in inv]
    lines += ["    const pops::Real %s = %s;" % (p, _recip_rewrite(e, inv_set).to_cpp())
              for p, e in items]
    return lines


def _jac_entries(model: Any) -> list:
    """Entries (Expr) of every ranked Jacobian sub-block (wave_speeds 'numeric'
    path). Drives the dead-code elimination of max_wave_speed / wave_speeds."""
    ws = model._ws_jacobian
    out = []
    axes = _ranked_axes(model)
    if tuple(ws["blocks"]) != axes or tuple(ws["rows"]) != axes:
        raise ValueError("wave-speed Jacobian must cover the exact emitted axis set %s" % (axes,))
    for key in axes:
        rows = ws["rows"][key]
        for b in ws["blocks"][key]:
            for gi in b:
                for gj in b:
                    out.append(rows[gi][gj])
    return out
