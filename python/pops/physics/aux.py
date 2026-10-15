"""Physical roles and runtime-parameter helpers.

Auxiliary identity, contracts and storage are owned solely by ``ProviderPack``.
"""
from __future__ import annotations

from typing import Any

__all__ = ["max_runtime_params", "role_of", "roles_for"]


def max_runtime_params() -> None:
    """There is no model-specific parameter-count ceiling.

    Bind authenticates exact cardinality and an explicit configurable byte
    capacity for owned host/device storage. Native index representation remains
    checked independently when emitting package metadata.
    """
    return None


# --- Physical roles: variable name -> exact structured token ---------------
# CANONICAL mapping name -> physical role token. Lets a
# generated brick DECLARE the MEANING of its components (density, momentum, energy...) instead of
# empty roles, so that inter-species couplings (System::add_collision / add_thermal_exchange)
# resolve via index_of(role) rather than via a literal index. The usual names of fluid models
# (rho, rho_u, u, p, E, n...) are recognized; an unknown name becomes an exact user-role label. A
# model can impose
# its roles explicitly with typed ComponentRole descriptors
# (conservative_vars(..., roles=[...]) / set_primitive_state(..., roles=[...]))
# for a non-standard layout. Key = exact variable name, value = structured native token.
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
    """Canonical physical token or exact custom label inferred from ``name``.

    Unknown components keep their identity (``q1`` and ``q2`` do not collapse
    onto one anonymous ``custom`` token).  The typed :class:`Custom` descriptor
    owns validation of labels that cross the native metadata boundary.
    """
    physical = CANONICAL_ROLES.get(name)
    if physical is not None:
        return physical
    from .roles import Custom, native_role_token

    return native_role_token(Custom(name))


def roles_for(names: Any, override: Any = None, *, dimension: Any = None) -> Any:
    """Lower typed role descriptors to exact native tokens parallel to ``names``.

    ``None`` entries request canonical name inference.  This function is the sole
    authoring-to-token boundary: role strings are rejected instead of being treated
    as a compatibility vocabulary.
    """
    names = tuple(names)
    if override is None:
        values = (None,) * len(names)
    else:
        if isinstance(override, (str, bytes)):
            raise TypeError("roles must be an ordered iterable, not a string")
        try:
            values = tuple(override)
        except TypeError:
            raise TypeError("roles must be an ordered iterable") from None
        if len(values) != len(names):
            raise ValueError("roles: %d roles for %d variables" % (len(values), len(names)))
    from .roles import ComponentRole, native_role_token, parse_role

    lowered = []
    for index, (name, role) in enumerate(zip(names, values, strict=True)):
        if role is None:
            lowered.append(role_of(name))
            continue
        if isinstance(role, str):
            raise TypeError(
                "role %d requires a typed ComponentRole, not a string" % index
            )
        if not isinstance(role, ComponentRole):
            raise TypeError("role %d must implement ComponentRole" % index)
        lowered.append(native_role_token(role, dimension=dimension))

    seen: dict[str, int] = {}
    for index, token in enumerate(lowered):
        parsed = parse_role(token, dimension=dimension, where="state role %d" % index)
        previous = seen.get(parsed.token)
        if previous is not None:
            raise ValueError(
                "state roles declare duplicate token %r at components %d and %d"
                % (parsed.token, previous, index)
            )
        seen[parsed.token] = index
    return lowered
