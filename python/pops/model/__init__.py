"""Operator-first type system (Spec 2, phase S2-1).

This package defines the abstract spaces and typed operators that a model-free
``pops.time.Program`` composes:

* ``StateSpace`` -- a conservative/primitive state space (the components of ``U``);
* ``FieldSpace`` -- an auxiliary or solved-field space (e.g. ``phi, grad_x, grad_y``);
* ``RateSpace`` / ``Rate(U)`` -- the tangent of a ``StateSpace`` (``dU/dt``);
* ``LocalLinearOperator(U, U)`` / ``MatrixFreeOperator`` -- operator-valued types;
* ``Signature`` -- a typed ``(inputs) -> output`` contract;
* ``Operator`` and ``OperatorRegistry`` -- a named, typed, integer-id'd registry.

These types are a TYPED VIEW: they carry no numerics and no array data. In phase
S2-1 the registry is DERIVED from an existing :class:`pops.dsl` model -- the PDE
shortcuts ``source_term`` / ``linear_source`` / ``elliptic_field`` / ``flux`` lower
into typed operators without changing the public PDE API. The public
``pops.model.Module`` front-end (S2-3), the typed ``P.call`` (S2-2) and the C++
codegen consumption (S2-6) build on these primitives in later phases.

The package imports only the standard library so it can be exercised without the
compiled ``_pops`` extension.
"""
from .bundles import RateBundle
from .handles import Handle, OperatorHandle, OwnerPath, ParamHandle, StateHandle
from .ownership import (
    AmbiguousReferenceError,
    DoubleOwnershipError,
    IdentityCollisionError,
    MissingOwnershipError,
    OwnerKind,
    OwnerSegment,
    OwnershipError,
    UnresolvedOwnershipError,
)
from .manifest import (
    ModuleManifest,
    OperatorManifestEntry,
    OperatorRegistryManifest,
    build_module_manifest,
)
from .module import Module
from .operators import (
    OPERATOR_FAMILIES,
    OPERATOR_KINDS,
    OPERATOR_REQUIREMENT_KEYS,
    OPERATOR_SIGNATURE_CONTRACTS,
    LocalLinearOperator,
    MatrixFreeOperator,
    Operator,
    SignatureContract,
    operator_family,
    validate_operator_signature,
)
from .param_registry import ParamRegistry
from .bind_schema import BIND_SCHEMA_VERSION, BindSchema, BindSlot, ResolvedBindings
from .registry import DeclarationIndex, OperatorRegistry
from .component_protocols import (
    Effects,
    FallibleEvaluation,
    Format,
    Lowering,
    Provider,
    Report,
    Requirement,
    Restart,
    Stability,
    Stencil,
)
from .component_adapters import (
    ComponentAdapter,
    ComponentInterfaceError,
    ComponentProvenance,
    EvaluationOutcome,
    InterfaceBinding,
    InterfaceSpec,
    adapt_component,
)
from .component_registry import (
    ComponentRecord,
    ComponentRegistry,
    ComponentRegistrySnapshot,
)
from ._component_manifest import (
    ComponentExtensionSchema,
    ComponentManifest,
    ComponentManifestError,
    ComponentVersion,
)
from ._generated_component_schema import COMPONENT_MANIFEST_SCHEMA_VERSION
from .provider_pack import (
    ComponentContract,
    ComponentKey,
    MissingInputProvider,
    ProviderEntry,
    ProviderPack,
    build_operator_provider_pack,
    build_provider_pack,
)
from .signatures import Signature
from .spaces import (
    AuxSpace,
    FieldSpace,
    Rate,
    RateSpace,
    Space,
    StateSpace,
)

__all__ = [
    "Handle", "StateHandle", "ParamHandle", "OperatorHandle", "OwnerPath", "OwnerKind",
    "OwnerSegment",
    "OwnershipError", "MissingOwnershipError", "DoubleOwnershipError",
    "AmbiguousReferenceError", "IdentityCollisionError",
    "UnresolvedOwnershipError",
    "ModuleManifest",
    "OperatorManifestEntry",
    "OperatorRegistryManifest",
    "build_module_manifest",
    "Requirement", "Lowering", "Stencil", "Stability", "Provider", "Effects",
    "Restart", "Report", "FallibleEvaluation", "Format",
    "ComponentAdapter", "ComponentInterfaceError", "ComponentProvenance",
    "EvaluationOutcome", "InterfaceBinding", "InterfaceSpec", "adapt_component",
    "COMPONENT_MANIFEST_SCHEMA_VERSION", "ComponentExtensionSchema", "ComponentManifest",
    "ComponentManifestError", "ComponentVersion", "ComponentRecord",
    "ComponentRegistry", "ComponentRegistrySnapshot",
    "ComponentKey", "ComponentContract", "ProviderEntry", "ProviderPack",
    "MissingInputProvider", "build_provider_pack", "build_operator_provider_pack",
]
