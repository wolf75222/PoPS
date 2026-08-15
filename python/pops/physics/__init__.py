"""Physical-model authoring through the single canonical :class:`Model` facade.

The implementation engines remain in their private modules for lowering existing operator
registries; they are not alternate authoring APIs. Generic composition happens through typed
operators and small protocols returned by ``Model``.
"""

from .board import Model
from .admissibility import (
    AdmissibilityConstraint,
    AdmissibleSet,
    ConstraintKind,
    EnforcementPhase,
    EnforcementRule,
    EnforcementSchedule,
    ProjectionProvider,
)
from .inversion import (
    InversionProviderCatalog,
    InversionWorkspaceBudget,
    PreparedInversionProvider,
    VariableInversionProblem,
)
from .roles import (
    Axial,
    ComponentRole,
    Custom,
    Density,
    Energy,
    Momentum,
    Pressure,
    RoleKey,
    Scalar,
    StateSchema,
    Temperature,
    Velocity,
)

__all__ = [
    "AdmissibilityConstraint", "AdmissibleSet", "ConstraintKind", "EnforcementPhase",
    "EnforcementRule", "EnforcementSchedule", "InversionProviderCatalog",
    "InversionWorkspaceBudget", "Model", "PreparedInversionProvider", "ProjectionProvider",
    "VariableInversionProblem", "Axial", "ComponentRole", "Custom", "Density", "Energy",
    "Momentum", "Pressure", "RoleKey", "Scalar", "StateSchema", "Temperature", "Velocity",
]
