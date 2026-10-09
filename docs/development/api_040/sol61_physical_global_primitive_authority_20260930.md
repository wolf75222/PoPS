# Detached physical globals: constitutive recipe authority

Date: 2026-09-30. Parent: `2fbf709a57e8df1192279bf6dab669636b048edc`.

The independent reviewer found that the parent authenticated the registered
source body but expanded mutable `prim_defs` afterwards. In a public model,
`loss = .47 * charge` and `source = -loss * state` therefore admitted a different
lowered `loss` under an unchanged Module hash. This was a source authority defect,
not a native numerical failure. The parent alone does not close that route.

Before expansion, provider planning, runtime slot assignment, or kernel emission,
`program_source_authority.py` now reconstructs the lowering from the authenticated
Module and the evaluation's exact StateSpace. It compares the source body, its
transitive primitive recipes and their consumed declaration order, conservative
component order, and the branch that selects primitive inlining. Every runtime
parameter read in the source closure must retain its registry-issued declaration,
default literal and dtype. The model-wide sorted runtime parameter layout is also
checked because an extra parameter elsewhere can shift the source's native slot.

The guard does not freeze or fingerprint the whole mutable compiler model. An
unconsumed cache and a dead constant primitive remain admissible and leave emitted
C++ unchanged. It retains no Case, authoring Model or registry in the detached
proof; the reconstructed lowering and parameter comparisons are temporary.
Existing live-issued global port, row, capture point, units, scope, input and body
checks remain in force. No header, runtime method, tolerance or physical equation
is changed.

The new public source fixture declares three distinct conservative coordinates,
two nested constitutive primitives, a true global port and, separately, a runtime
parameter. It resolves and detaches the actual Program. Mutating either primitive,
removing the transitive definition, reversing component routing, changing a
runtime default or declaration handle, or introducing a lowering-only slot is
refused before emission. A public `pops.compile(resolved_plan)` probe substitutes
only external compiler/toolchain access and stops at the actual `problem.cpp`
compiler seam. Its model lowering, proof preparation, detach and emission are
the real implementation; its placeholder model file is not a native artifact.

Validation with explicit source `PYTHONPATH=python` and the read-only
`pops-api040/bin/python`:

- 82 tests pass across primitive authority, resolved global authority, physical
  globals, integral candidate capture and Module lowering. The final dead-cache
  and dead-primitive check also passes separately after its last extension.
- Nine import-graph and codegen-without-NumPy architecture checks pass.
- Four historical IR/plan/Module-manifest tuples, covering legacy manifest 10
  and physical-global manifest 11, and eight complete System/AMR C++ hashes are
  byte-identical to the pinned parent of the original authority correction.
- Ruff passes on all three changed Python files, and Git diff whitespace checks
  pass.
- The new full System and AMR three-component runtime-parameter translation units
  are checked with `/usr/bin/clang++ -std=c++20 -fsyntax-only`, native Dim2 and
  the actual read-only MPI/Kokkos headers; both pass. No compiler installation,
  JIT or package installation is used.

This is source and host syntax evidence. Root owns installed package rebuild and
the true Serial/MPI physical-global executions. No native solution, MPI vote,
GPU execution or end-to-end scientific reception is inferred from these checks.
