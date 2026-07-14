"""pops.params -- typed scalar / small-object parameters (Spec 5 sec.5.12).

A parameter declares whether it is compile-time (:class:`ConstParam`) or runtime
(:class:`RuntimeParam`), its typed dtype (:mod:`pops.math`), an optional default and an
optional typed :mod:`~pops.params.constraints` domain -- instead of the string form
``Param(kind="runtime")`` / ``domain="positive"``. The canonical constant alias lives in
:mod:`pops.params.constants`. Everything here is inert; the codegen / runtime consume the
descriptors (a runtime param appears in ``compiled.arguments()``; a const param is in the
cache key).
"""
from .runtime import (
    MISSING,
    PARAM_DECLARATION_SCHEMA_VERSION,
    ConstParam,
    DerivedParam,
    ParamDefaultState,
    ParamInvalidation,
    ParamKind,
    ParamPhase,
    ParamProvenance,
    ParamStorage,
    ParameterDeclaration,
    RuntimeParam,
    validate_parameter_data,
)
from .use_sites import (
    InvalidParamUseSite,
    PARAM_USE_MATRIX,
    ParamUse,
    resolve_param_use,
    validate_param_use,
)
from .constraints import Constraint, Positive, NonNegative, Range, In, Interval, OneOf
from .constants import Constant
from . import constraints, constants

__all__ = [
    "MISSING", "PARAM_DECLARATION_SCHEMA_VERSION",
    "ParameterDeclaration", "ParamKind", "ParamStorage", "ParamPhase",
    "ParamInvalidation", "ParamDefaultState", "ParamProvenance",
    "RuntimeParam", "ConstParam", "DerivedParam",
    "validate_parameter_data",
    "InvalidParamUseSite", "PARAM_USE_MATRIX", "ParamUse",
    "resolve_param_use", "validate_param_use",
    "Constraint", "Positive", "NonNegative", "Range", "In", "Interval", "OneOf",
    "Constant", "constraints", "constants",
]
