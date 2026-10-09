# Public Python library aliases, version 1

`pops.public_api_exports.PUBLIC_LIBRARY_ALIAS_VERSION = 1` defines a compatibility
route to an existing Python library class. A frozen `PublicLibraryAlias` records
its public module/name, canonical module/name, repository source path, export
kind and contract version. The registry is a read-only mapping. Unknown names,
unsupported versions, non-library owners, non-class values and mismatched source
owners fail closed.

The existing `pops.numerics.FanLi15RawMomentPath` export remains in `__all__` and
`dir(pops.numerics)`. Lookup returns exactly
`pops.moments.fan_li_path.FanLi15RawMomentPath`, including its existing class
identity. It does not construct a path or receive physical expressions, state,
method inputs or numerical parameters. Constitutive expressions, coefficients,
guards and serialization remain in their existing Python library owner.

The generic `PathArithmeticComposition` marker belongs to `_ir.path_arithmetic`;
`numerics.normalized_polynomial_path` retains an identity-preserving re-export.
The marker has no model definitions or imports. The `Handle` reference in the
constitutive library is an annotation imported under `TYPE_CHECKING`.

Architecture distinguishes this exact registered class route from imperative
imports. Its AST fence accepts only the resolver call with the current module
identity and attribute name. Unregistered hooks, class construction, alternate
imports, overwritten module identity and missing public exports are rejected.
The numerics package explicitly consumes the lower `_ir` language primitive; a
Source fence pins its exact generic marker import and the existing `_ir` sink
policy. All other ordinary package permissions remain unchanged, including the
refusal of an imperative numerics-to-moments import. The contract module imports
standard-library modules only.

This contract provides public API compatibility, not compiler recognition of a
model identity. The accompanying lower-owner moves preserve traversal and checkpoint-origin
semantics; phase checks use exact typed component contracts. No numerical guard,
acceptance criterion or physical mathematical body changes. Source tests do not establish native CPU,
MPI, CUDA or full language qualification.

The numerical library reads typed expressions through `model.expression_language@1`.
This lower declaration surface re-exports the existing `_ir` primitives and transforms
with exact object identity. It contains no functions, classes, model definitions or
numerical recipes. All lexical numerics-to-`_ir` imports (absolute, relative, nested,
and literal dynamic routes) are refused except the exact module-scope path marker.
