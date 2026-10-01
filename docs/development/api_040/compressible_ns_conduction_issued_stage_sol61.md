# Periodic compressible NS with conduction: prepared source fixture

This is a new, explicitly chosen nondimensional benchmark, not a claim that the
missing Couette wall or other M05 corpus inputs have been supplied. The corpus
requires physical viscous stress and heat conduction in the compressible energy
balance. This witness declares a periodic interval `[0,1]`, ideal-gas parameters
`gamma=1.4`, `R=1`, `Cv=2.5`, `mu=.03`, and `kappa=.02`.

The physical declarations precede the numerical and temporal choices in
`tests/python/support/ns_conduction.py`. Conserved `Q=(rho,rhou,rhoE)` and physical
unknowns `Psi=(rho,u,T)` are distinct products. The accumulation is
`H(Psi)=(rho,rho*u,rho*(Cv*T+u*u/2))`. Euler's authored physical flux includes the
pressure and its actual characteristic waves. The viscous and thermal rate is
`(0, div((4*mu/3)*grad(u)), div((4*mu/3)*u*grad(u)+kappa*grad(T)))`.
The resulting diffusion matrix is singular and nonsymmetric; no SPD assertion
is made. Its energy/velocity entry depends on the current unknown `u`.

The selected method is one explicit FirstOrder/Rusanov Euler step followed by an
original implicit stage:

```
Qstar = Qn + dt * R_Euler(Qn)
F(Psi) = H(Psi) - tau * R_viscous_thermal(Psi) - Qstar = 0
tau = Program.temporal_tau(Program.dt, at=TimeState.next.point)
```

`Arithmetic@1/PerCandidate@1` evaluates the true coefficient at every `Psi` and
both finite-difference perturbations. The existing full original residual and
true correction residual checks remain mandatory. This changes neither equation
nor seven Newton controls: `1e-10,20,1e-8,240,60,1e-4,1/1024`, with FD step `1e-6`.
The fixed step is `.0002`. Positivity guards check density and temperature, rather
than imposing positivity on momentum. They are failure guards, not clipping.

## Two explicit generic ports

The old accepted-previous stage correctly refuses a computed future `Qstar`.
Commit `61886a2540c308efb1438a6a4032348d93277d91` adds
`previous_scope='issued'`, projection schema 3, the original stage contract @3,
and `pops.evolved-original-field-issued-previous@1`. It requires a current issued
expression at the exact official next TimePoint and authenticates all accepted
State-read ancestors. A relabeled `.n` read is refused. The capture remains part
of the immutable original SolveRequest and its equation/initialization identity.

Commit `b3f2003748a2c579083cef4d2ac99a8455760b51` adds
`SolveRequest.seed_product(program=..., expressions=(...), space=FieldSpace(...),
name=...)`, contract `pops.original-field.typed-product-seed@1` and conditional
IR21. The output is a physical scalar-field product with an exact ordered
unknown tuple, explicit units, point, issuer and problem identity. Its State
input supplies allocation/layout only; it supplies neither output component
width nor conservative physical representation. `state_ref` remains absent.
For NS, the genuine pointwise Kokkos kernel evaluates inverse EOS into `Psi`.
A second public admission test emits a width-two `(rho,mu)` guess on a width-one
State allocation, without changing the State or declaring a fake StateSpace.

Allocation and kernel exceptions are fenced and voted on the prepared execution
lane before the following numerical-status collective. Scratch remains an
attempt-owned resource; it is not a mutable model parameter or accepted state.
This port uses existing headers and runtime APIs. Native ABI/wire ordinals and
the header signature are unchanged by these Python commits. A new generated
program must still be compiled and linked by the genuine installed runner.

## Evidence and pending reception

Agent checks are **SOURCE_ONLY**. The coherent suite has 49 passing checks before
the additional mixed-width admission; its 34 historical stage checks are intact.
The additional genuine Case/validate/resolve/emit width-one to width-two admission
passes separately in 2.25 seconds. The final new suite's other 15 checks also pass.
Four fresh legacy profiles (IR12 constant/candidate, IR14 additive, IR15
partitioned) have exactly equal complete IR, C++, module hashes/manifests and
SolveRequests against checkout `72e74dda07f32d7e2e2f9febdb63919bf39ff5ad`.
Their joint-image hashes are respectively:

```
acfe45002312866550a98a5b4c8a607b1e1079200fe652e2cb5a07bcf4a251c5
405287dab6f6e276cf180b4dd0ad97d9455087285eecb9023250e5b192c2e766
007cda80d0483e33db27598185649eaa6a1b697c6090d0edf47fe036d906091b
f2823d0b82959fcb664fbf32f3519f262067131c1e6a110b7bba2f1f16083c4f
```

Source command (the interpreter is read-only; no native compilation):

```
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -c 'import sys; sys.path[:0]=["python","."]; import pytest; raise SystemExit(pytest.main(["tests/review/test_sol61_issued_original_previous.py","tests/review/test_sol61_evolved_original_field_stage.py","tests/review/test_sol61_evolved_stage_mms.py","-q","-p","no:cacheprovider"]))'
```

The future installed-native nodes are:

```
tests/python/integration/runtime/test_public_ns_conduction.py::test_public_compressible_ns_conduction_original_stage_exact_replay[8]
tests/python/integration/runtime/test_public_ns_conduction.py::test_public_compressible_ns_conduction_original_stage_exact_replay[16]
```

They use the real canonical BindArray IC subject, actual conservative cell means,
two executed steps, detached scalar and product histories, original solver
diagnostics, accepted auxiliary checkpoint bytes and clocks. They authenticate
each checkpoint immediately, preserve its hash, restart/replay exactly, and
verify creator/continuation provenance before comparing payload bytes. Actual
generated C++, carried IR, DSO and sidecars are archived from the compiled
component. All MPI calls use the existing collective helpers.

Independent face sums reconstructed from the saved actual fields verify the
original stage, periodic mass/momentum/total-energy conservation, accumulation,
stress work and heat conduction with a predeclared `3e-8` guard. Those arrays are
explicitly named `COMPUTED_REFERENCE`, not passed off as native face getters.
Positive constitutive entropy production is separated from Rusanov numerical
dissipation; diffusion redistributes total energy without a second artificial
energy subtraction.

No Native compilation, execution, MPI failure/rollback injection, convergence
study, Couette-wall case, general AMR NS, GPU or complete M05 qualification has
been received here. Root owns the new installed artifact and serial/MPI runs.
