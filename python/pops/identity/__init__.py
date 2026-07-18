"""Canonical cross-language identity primitives."""

from .digest import Identity, make_identity
from .encoding import canonical_bytes, canonical_sha256
from .scalar import (
    CPP_INT_MAX,
    ScalarLiteral,
    exact_cpp_int,
    exact_numeric_scalar,
    scalar_cpp,
    scalar_data,
)
from .artifact import (
    artifact_identity, artifact_spec_identity, binary_bundle_identity, binary_identity,
)
from .semantic import (
    SEMANTIC_SCHEMA_VERSION,
    model_semantic_data,
    program_semantic_data,
    semantic_identity,
    semantic_identity_of,
)

__all__ = [
    "Identity", "artifact_identity", "artifact_spec_identity", "binary_bundle_identity",
    "binary_identity",
    "CPP_INT_MAX", "ScalarLiteral",
    "canonical_bytes", "canonical_sha256", "exact_cpp_int", "exact_numeric_scalar",
    "make_identity", "model_semantic_data", "scalar_cpp", "scalar_data",
    "program_semantic_data", "semantic_identity", "semantic_identity_of",
    "SEMANTIC_SCHEMA_VERSION",
]
