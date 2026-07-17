"""Public operator-first model SDK.

This package defines the abstract spaces and typed operators that a model-free
``pops.time.Program`` composes:

* ``StateSpace`` -- a conservative/primitive state space (the components of ``U``);
* ``FieldSpace`` -- an auxiliary or solved-field space (e.g. ``phi, grad_x, grad_y``);
* ``RateSpace`` / ``Rate(U)`` -- the tangent of a ``StateSpace`` (``dU/dt``);
* ``LocalLinearOperator(U, U)`` / ``MatrixFreeOperator`` -- operator-valued types;
* ``Signature`` -- a typed ``(inputs) -> output`` contract;
* ``Operator`` and ``OperatorRegistry`` -- a named, typed, integer-id'd registry.

These types carry declarations, signatures and identities, never numerical arrays. ``Module`` is
the canonical compiler-facing model authority; the public physics ``Model`` facade builds the same
typed spaces and operators. ``Program`` refers to them through qualified handles and its single
``solve``/operator-call contracts.

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
    "Module", "RateBundle",
    "Space", "StateSpace", "FieldSpace", "AuxSpace", "RateSpace", "Rate",
    "Signature", "SignatureContract", "Operator", "LocalLinearOperator",
    "MatrixFreeOperator", "OperatorRegistry", "DeclarationIndex", "ParamRegistry",
    "OPERATOR_FAMILIES", "OPERATOR_KINDS", "OPERATOR_REQUIREMENT_KEYS",
    "OPERATOR_SIGNATURE_CONTRACTS", "operator_family", "validate_operator_signature",
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
