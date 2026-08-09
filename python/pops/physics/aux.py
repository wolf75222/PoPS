"""Generic auxiliary declaration helpers, physical roles, and runtime parameters.

The exact owner/space/component/slot authority is ``ProviderPack``.  The
small detached ``AuxLayout`` helper is only an ordered name view for Python
diagnostics; it has no dimension or physical-name convention.
"""
from __future__ import annotations

from typing import Any

from pops._aux_layout import AuxLayout, aux_component_index, aux_layout, aux_total_n_aux

# Bound on the number of RUNTIME parameters per block (P7-b). MIRROR of kMaxRuntimeParams
# (include/pops/runtime/config/runtime_params.hpp): the C++ carrier RuntimeParams has an array of this
# FIXED size (device-copiable without allocation), so a model exceeding the bound is rejected at codegen.
# This module stays stdlib-only (no _pops import), so the value is a literal; _pops exposes the same
from pops._native_facts import NATIVE_MAX_RUNTIME_PARAMS

__all__ = [
    "AuxLayout",
    "aux_component_index",
    "aux_layout",
    "aux_total_n_aux",
    "max_runtime_params",
    "role_of",
    "roles_for",
]

_K_MAX_RUNTIME_PARAMS = NATIVE_MAX_RUNTIME_PARAMS


def max_runtime_params() -> int:
    """Return the declared release-provider capacity without loading a native runtime.

    The installed-extension conformance gate independently proves that this generated/declared
    fact equals ``pops::kMaxRuntimeParams``. Authoring and validation therefore stay pure Python.
    """
    return _K_MAX_RUNTIME_PARAMS


# --- Physical roles: variable name -> VariableRole -------------------------
# CANONICAL mapping name -> physical role (cf. pops::VariableRole / role_name on the C++ side). Lets a
# generated brick DECLARE the MEANING of its components (density, momentum, energy...) instead of
# empty roles, so that inter-species couplings (System::add_collision / add_thermal_exchange)
# resolve via index_of(role) rather than via a literal index. The usual names of fluid models
# (rho, rho_u, u, p, E, n...) are recognized; an unknown name stays 'Custom'. A model can impose
# its roles explicitly (conservative_vars(..., roles=[...]) / set_primitive_state(..., roles=[...]))
# for a non-standard layout. Key = EXACT variable name, value = member of pops::VariableRole.
CANONICAL_ROLES = {
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


def role_of(name: Any) -> Any:
    """CANONICAL physical role of name @p name (member of pops::VariableRole), 'Custom' if unknown."""
    return CANONICAL_ROLES.get(name, "custom")


def roles_for(names: Any, override: Any = None) -> Any:
    """List of roles (pops::VariableRole members) parallel to @p names. @p override (optional):
    list of the same length explicitly fixing the roles (string 'Density'... or None to fall back
    on the canonical mapping of the name). Used for non-standard layouts where names are not enough."""
    if override is None:
        return [role_of(nm) for nm in names]
    if len(override) != len(names):
        raise ValueError("roles: %d roles for %d variables" % (len(override), len(names)))
    return [(r if r is not None else role_of(nm)) for nm, r in zip(names, override, strict=True)]
