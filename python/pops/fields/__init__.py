"""Final field-operator contracts: physics, numerics and materialization stay separate."""

from __future__ import annotations

from .context import (
    Accepted,
    FieldContext,
    FieldInput,
    FieldMaterialization,
    FieldValidity,
    LayoutBinding,
    Provisional,
    RecomputeField,
    UseHeldField,
    UseMaterializedField,
)
from .discretization import (
    CompositeHierarchySolve,
    FieldDiscretization,
    FieldHierarchyPolicy,
    InferHierarchyFromLayout,
    LevelByLevelSolve,
)
from .gauges import FieldGauge, MeanValueGauge, PinnedValueGauge
from .nullspace import ConstantNullspace
from .operator import FieldOperator
from .outputs import DerivedField, FieldOutput, GradientOutput
from .poisson import (
    AnisotropicPoissonOperator,
    PoissonOperator,
    ScreenedPoissonOperator,
)
from .policies import (
    FailFieldRead,
    FieldAttemptRejected,
    FieldConsumer,
    FieldFailureAction,
    FieldReadError,
    FieldReadPolicy,
    HoldLastValue,
    RecomputeAtDiagnostic,
    RecomputeAtOutput,
    RecomputeAtTagging,
    RejectFieldAttempt,
)
from . import aux, bcs, coefficients, gauges, nullspace, outputs, policies, rhs
from .catalog import fields as catalog


__all__ = [
    "Accepted",
    "AnisotropicPoissonOperator",
    "CompositeHierarchySolve",
    "ConstantNullspace",
    "DerivedField",
    "FailFieldRead",
    "FieldAttemptRejected",
    "FieldConsumer",
    "FieldContext",
    "FieldDiscretization",
    "FieldFailureAction",
    "FieldGauge",
    "FieldHierarchyPolicy",
    "FieldInput",
    "FieldMaterialization",
    "FieldOperator",
    "FieldOutput",
    "FieldReadError",
    "FieldReadPolicy",
    "FieldValidity",
    "GradientOutput",
    "HoldLastValue",
    "InferHierarchyFromLayout",
    "LayoutBinding",
    "LevelByLevelSolve",
    "MeanValueGauge",
    "PinnedValueGauge",
    "PoissonOperator",
    "Provisional",
    "RecomputeAtDiagnostic",
    "RecomputeAtOutput",
    "RecomputeAtTagging",
    "RecomputeField",
    "RejectFieldAttempt",
    "ScreenedPoissonOperator",
    "UseHeldField",
    "UseMaterializedField",
    "aux",
    "bcs",
    "catalog",
    "coefficients",
    "gauges",
    "nullspace",
    "outputs",
    "policies",
    "rhs",
]
