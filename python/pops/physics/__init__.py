"""pops.physics : math/physics model AUTHORING layer.

The public surface is the blackboard facade ``pops.physics.Model`` plus typed
parameter and aux/role helpers. Lower-level symbolic/codegen engines remain in their explicit
modules (``pops.physics.model``, ``pops.physics.facade``, ``pops.physics.hybrid``) and are not
re-exported here; ``physics`` is an authoring layer, not a compilation facade.
"""
# Aux-channel layout + physical roles (single Python-side source; mirror of the C++ headers).
from .aux import (
    AUX_CANONICAL, AUX_BASE_COMPS, AUX_NAMED_BASE, AUX_NAMED_MAX, CANONICAL_ROLES,
    aux_n_aux, aux_total_n_aux, role_of, roles_for)

# Typed model parameters used by physics / case authoring. The internal carrier exists in
# pops.physics.model, but ``Param(kind=...)`` is rejected there too; mode selection is only through
# typed constructors.
from .model import RuntimeParam, ConstParam

# Generic coupled inter-species source authoring (compiled handles stay in the module).
from .multispecies import CoupledSource

# Blackboard board facade: the public pops.physics.Model surface (Spec 3).
from .board import Model
# Spec 5 sec.5.16 / sec.11 preferred name. ALIAS, not a rename: it is the SAME class object
# (PhysicsModel is Model), so every existing pops.physics.Model consumer keeps working and the
# class __name__ stays "Model" (a `type(x).__name__ == "Model"` check is unaffected).
PhysicsModel = Model
from .board_handles import (
    Invariant, FluxHandle, SourceHandle, FieldsHandle, FieldHandle,
    LocalLinearOperatorExpr, CallableOperator, StateHandle, VectorHandle,
    _roles_for)  # restore the flat physics.py module-level access (test_riemann_capabilities)

__all__ = [
    # board surface (the historical pops.physics public names)
    "Model", "PhysicsModel", "Invariant", "FluxHandle", "SourceHandle", "FieldsHandle", "FieldHandle",
    "LocalLinearOperatorExpr", "CallableOperator", "StateHandle", "VectorHandle",
    # aux + roles
    "AUX_CANONICAL", "AUX_BASE_COMPS", "AUX_NAMED_BASE", "AUX_NAMED_MAX", "CANONICAL_ROLES",
    "aux_n_aux", "aux_total_n_aux", "role_of", "roles_for",
    # typed parameters
    "RuntimeParam", "ConstParam",
    # coupled source
    "CoupledSource",
]
