# AMR GMRES residual authentication and reorthogonalization

The authentic typed diagnostic identifies iteration-budget exhaustion at the
first Newton iteration of the N32, component-order 012 original witness. It
completes 240 columns at restart 80; the explicitly recomputed linear norm is
`0x1.05538b6317a3ap-12`, exceeding the requested stop
`0x1.6f18ebf6991e7p-14`. The projected norm closely agrees. The preceding two N16
subcases complete before this failure. This is not a late-Newton cancellation
floor or a failure confined to MPI. The non-preconditioned failure remains an
authentic historical result, not a reason to increase the declared budget.

A separate independent counterexample established that the old projected-stop
branch could return linear convergence without evaluating the actual JVP on
the complete correction. For the finite SPD matrix
`[[64000000000000.36,-47999999999999.52],
  [-47999999999999.52,36000000000000.64]]`, RHS `(.3,.7)` and relative tolerance
1e-5, the projected residual can be zero while the true residual exceeds the
stop. This ill-conditioned example is distinct from the moderate original AMR
witness and does not identify its cause.

The workspace now performs two modified Gram--Schmidt passes per Arnoldi
column, accumulating both projection coefficients. A projected stop ends the
current Arnoldi cycle, but only a finite explicitly recomputed
`rhs - JVP(complete correction)` can authorize linear convergence. If this
recheck misses the stop, restart continues within the original remaining budget.
Neither the equation, finite-difference step, norm/coverage, solver family,
Newton/linear budgets, restart, line-search controls nor tolerance changes.
The extra pass adds dot products and vector updates; successful projected stops
now also incur the previously omitted full-correction JVP.

Actual extracted GMRES/update bodies are compiled over scalar vector algebra
for identity widths 1/3/5, a direction-dependent approximate JVP counterexample,
and the independent SPD counterexample. The host seams implement no PDE or AMR
runtime. The old early-return mutant fails the complete-correction invocation
check. The source proof for the earlier diagnostic-only freeze is explicitly
pinned to its original SHA rather than falsely asserting that this numerical
change leaves its operations byte-identical. Full actual native-test TU syntax
passes with real GoogleTest/Kokkos/MPI headers. Native acceptance, including
whether MGS2 changes the N32 outcome, remains root-owned and pending.

An independent source-derived N32 matrix predicts that MGS2 alone still exceeds
the declared budget. A preconditioned realization, if implemented and explicitly
selected, requires a distinct reception and must preserve the non-preconditioned
history. It cannot be presented as an unchanged-method positive result.
