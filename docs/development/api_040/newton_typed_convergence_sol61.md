# Explicit Original Newton stopping policy — Source preparation

Base: 5486e2d9. No Native extension, runtime, MPI execution or JIT qualification is claimed.

`Newton(tolerance=Relative(1e-10, floor=AbsoluteFloor(1e-11)))` selects
`norm(FOriginal) <= max(1e-11, 1e-10 * norm(FOriginal_initial))`.
`Relative` without a floor uses precisely `rel * reference`, including zero reference;
`Absolute` uses its absolute coefficient. Numeric tolerance and the default retain
`tolerance * max(1, reference)` and the seven historical controls.
All coefficients have a closed canonical finite binary64 representation. Bool, raw
scalar encodings, unknown fields/kinds/contracts, negative coefficients and noncanonical
spelling are refused. This selects a stopping policy, not a different residual or norm.
Uniform keeps its DOF L2; AMR keeps the volume-weighted finest-active L2.
Raw relative residual remains diagnostic and uses the historical zero-reference denominator.

The shared public header helper `field_newton_stop_tolerance` is used by both
Newton-Krylov workspaces and the AMR Original recheck. Uniform emission uses it only
for the explicit policy, preserving historical generated C++ bytes. No residual
application or collective reduction is added. Complete Original equation bodies,
unknown order, captures and existing publication guards are retained.

Prepared spatial explicit-policy identity is `prepared-spatial-newton-v4`; its payload
pins the complete legacy solver identity and the closed convergence contract
`pops.newton.original-residual-convergence@1`. ProgramIR22 is conditional on this
policy (the base maximum is21). Installed `PreparedFieldNonlinear` uses schema2 only
for the explicit policy; schema1 and the legacy install entry remain unchanged.
The public `set_field_newton_convergence_plan` carries all seven controls and typed
coefficients together, validates before publication, and refuses a legacy policy.
No builtin field-provider spelling switch was added. Existing installed providers
still supply their complete declared field residual; this does not qualify them as
Program Original coupled-equation witnesses. Implicit accumulation stages currently
refuse typed convergence explicitly instead of silently ignoring it.

## ABI integration reserved to ROOT

`FieldNewtonOptions` layout and two exported runtime entry inventories change.
ROOT must integrate the pending Halo ABI7 tranche first and issue central NativeABI8
with regenerated release products, capability/header signatures, SDK and extension.
This author deliberately leaves all central release constants unchanged. The new
installed protocol fails closed when the method is absent. Header/package/DSO byte
identity and fresh Original uniform/AMR Native qualification are required before use.
Checkpoint payload12 is unchanged; explicit policy is authenticated by program/provider
and exact runtime contracts. Historical reader qualifications are not extended by this work.

## Reproduction

Use `env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops/bin/python`, insert
this worktree's absolute `python` path, and run pytest for
`tests/review/test_sol61_newton_typed_convergence.py`,
`tests/python/unit/fields/test_nonlinear_mixed_field_problem.py`, and
`tests/python/unit/fields/test_amr_original_field_codegen.py`.
The first test asserts the actual Source `pops.__file__` and absence of `_pops`.
The host probe compiles the actual public header against existing Kokkos headers.

Legacy parity was checked in separate interpreters using exactly the same physical
fixture file (its source provenance is therefore equal): default options, both prepared
identities/manifests, complete ProgramIR and emitted C++ were byte-identical. The
companion JSON pins the combined comparison. Source checks are not GitHub CI.

The actual repository `test_numerical_defaults.cpp` adds a gtest case. Its small private
CPU build reuses the repository CMake-generated compile/link commands and already-built
read-only gtest/test-main dependencies. Commands/logs live in `build-newton-typed-host`;
no installed package or shared Native checkout is written. This is a host C++ check,
not an installed Python Native Original simulation.

Final validation: existing Original Source suites30 PASS121.16s; author policy suite23 PASS (see gel report for duration); real repository host gtest5 PASS, zero failures/skips. The host companion JSON pins the actual source, binary, XML and commands. No Native simulation acceptance is inferred.
