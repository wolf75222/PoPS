# Independent M05 periodic shear review

Reviewed Sol's `23b6c218`, `bfa53e80`, `5cbc487f`, `76984295` as integrated
through `25eaef9a`/`15a776c6`. No production-core change or scientific
criterion change is proposed by this follow-up.

## Mathematical scope and oracle

The preserved `reference/PoPS_API_v0.4.0/results/corpus_matrix.json`, M05
lines 96–113, explicitly names the closed scalar periodic subproblem
u_t = nu u_xx, nu=.03, u(x,0)=sin(2*pi*x), N=32/64/128, t=.1. It also
states that full Couette/conducting Navier–Stokes needs thermal data.
The example follows precisely that closed subcase. It is not a new physical
model substituted for M05, and it does not qualify the complete family.
The H02 legacy script checked the random-vector semidiscrete dissipation
identity only; real public Diffusion evolution, three saved native states
per grid and accepted-face accounting are new evidence.

The oracle has no PoPS import. True sine cell means from sinc agree with
the independent antiderivative formula. For h=1/N its positive spatial
symbol is 4*N^2*sin(pi/N)^2 and the rate eigenvalue is its negative times
nu. Continuous, semidiscrete and N-step FE amplitudes are kept separate.
The prescribed dt=.1/N gives the incident-face bound 2*nu*N^2*dt=.768
at N128, below .9. FE is stable for this schedule, but spatial/time errors
can cancel; the receipt correctly makes no monotone-order claim.

For arbitrary periodic vectors the work h*u^T*L*u equals the negative
face-jump quadratic form. FE adds the positive dt^2*h*||L*u||^2/2 term
to the energy increment. The code tests that correction instead of calling
the whole FE energy drop physical dissipation. The native last-step
ledger uses two incidences per cell, signed flux nu*N*(u_right-u_left),
unit face measure and the actual accepted dt. Summed amounts reconstruct
h times the saved cell change, with zero global mass exchange.

An independent dense 17x17 periodic operator counter-test verifies the
rate, constant shifts, translations, reversal, work and FE energy formula
without using the production provider. It passed before any fixture edits.

## Reception defects found and repaired

1. The native rows were checked in memory but not saved. The NPZ/summary
   alone could not reproduce the exact observed ledger check later. The
   example now saves all raw rows to `ledger_N.json`, reopens that JSON
   before assessment, and records its SHA-256 alongside the NPZ hash.
   Example/oracle file hashes are also recorded. No synthetic ledger is
   substituted for the native diagnostic output.
2. Four receipt assertions in the pytest wrapper ran rank-locally before
   the following broadcast. They now run inside `collective_check` so a
   local assertion failure converges before the next collective. Native
   authoring, initial allocation, bind/run/gather/status are already
   separated into collective phases by the example.
3. `_ledger_metrics` did not explicitly reject nonfinite numeric fields.
   In particular `max(0, NaN)` can leave the flux error at zero, and
   `abs(NaN) > tolerance` is false for face measure or temporal weight.
   Four new counter-tests were **red** before the fix (1 independent
   mathematics test passed). All four numerical ledger fields now must be
   finite before metrics. A NaN integrated amount previously produced a
   NaN metric that the outer assessor rejected; its explicit early refusal
   is now consistent with the other fields. This is a reception-tool
   defect, not evidence that the native provider emitted nonfinite records.

No tolerances, PDE, physical parameters, grid, schedule, native operators,
or source/thermal interpretation were changed.

## Evidence boundary

After the follow-up: **18/18 pure/source tests pass** (13 existing plus
5 independent), Ruff and diff checks pass; the native three-grid test
collects once. No compilation/JIT/native execution was run by this review.

I independently inspected the parent's completed pre-follow-up receipts:
`installed-m05-periodic-shear` XML contains 14/14 (13 source/math plus the
one native three-grid test); `installed-m05-periodic-shear-mpi2` contains
1/1 on each rank with return code zero, unchanged installation/test flags
and exact rank testcase parity. Those receipts remain attached to their
original source and do not prove that this stronger checker or raw-ledger
archival ran. A central rerun is required for the new archived-ledger
evidence. Existing valid saved-state damping evidence is not relabelled
as a general qualification of the old permissive ledger checker.

The remaining complete M05 obligations include full stress/heat-flux
coupling, compressible/primitive gradient terms, thermal-energy accounting
and specified wall data. This periodic scalar run does not inherit any of
those claims, nor GPU, AMR, or remote HPC qualification.
