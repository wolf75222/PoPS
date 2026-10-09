# Original evolved field stage: additive source and partitioned Q

This is a mechanism witness for nonlinear accumulation and original field
stages, relevant to M06 and the M13 mechanism. It supplies no Marshak boundary,
material campaign, complete M13 data or native qualification.

## Production ports

`EvolvedOriginalFieldRate(spatial=operator, additive=f)` is the immutable
`pops.evolved-field-rate.spatial-additive@1` descriptor. It retains the exact
physical source expression. `spatial=None` or explicit zero declares an absent
spatial term; a pure captured additive source needs no fictitious reaction.
Supported additive expressions are literals and exact State captures. The RHS
port's unknown-dependent realization remains explicitly unsupported. This is a
realization scope, not a mathematical restriction on PoPS source laws.

The original equation is exactly

    Q(q+) - tau Rspatial(q+) = Qn + tau f(captures)

and its full original residual is rechecked before publication. Tau is
`program.temporal_tau(program.dt, at=Q.next.point)`, bound to the native issued
frame, never a literal requested dt. The local original residual compiler
already encodes captured RHS expressions and this Tau; no C++ header, ABI or
kernel change is required. The same sealed accumulation is projected with the
explicit `piecewise_constant_cell`, `cell_average`, `cell_volume` contract.
Descriptor identity projection revalidates supported dependencies. Additive
capture validation also traverses the complete input graph: a current native
State read must retain its accepted endpoint. A genuinely computed candidate
State at the declared solve point remains supported; relabeling an accepted
read as next, including through an enclosing expression, is refused.

Projection schema 2 discriminates the additive descriptor and conditionally
selects IR14. Legacy Stage remains schema 1 / IR12. FullLU's separate IR13 must
be merged with these feature tests using max priority, not replaced.

`pops.evolved-original-field-stage@2` permits separate physical Q State carriers
from one coupled solve. Exact qualified prior State/component quantities select
only their corresponding sealed accumulation expressions. Integer selection
indices, capture owners, spaces, blocks, regions, points, clocks, consumed
solution and original problem identity are rechecked. Program validation
requires every partition to be directly committed exactly once, with complete
coverage of that solved accumulation tuple. Existing transactional commit
publication/rollback owns the final atomic State update. Conditional IR15
supersedes IR14/13 for this partition. The single-carrier contract@1 and its
metadata stay unchanged. No wire or native ABI ordinal changes occur.

## Installed-native witnesses

The physical declarations precede the numerical stage/method and layout in
`tests/python/support/evolved_stage_mms.py`. Uniform periodic Cartesian 2D
N8/N16, dt=.01/.02 produce eight explicit cases:

* Scalar: H(T)=T+T² and D=.012.
* Two separate State carriers Q0/Q1 and three physical unknowns T0/T1/z:
  Q0=T0+T0²+.1 T1², Q1=T1+T1²+.2 T0 T1,
  z=.25 T0+.5 T1. The signed nonsymmetric cross-diffusion matrix is
  [[.012,.002],[-.001,.014]], with each column multiplied by
  1+.4 Tcolumn² under explicit PerCandidate@1 / Arithmetic@1.

Initial physical temperatures and first target are smooth, nonconstant and
fixed before execution. Qn is computed from those initial temperatures using
exactly the declared Q. The load is independently manufactured as

    f = (Q(Ttarget)-Q(Tinitial))/dt - div(D(Ttarget) grad Ttarget).

The NumPy oracle constructs oriented arithmetic face fluxes with real uniform
face lengths, differences them and divides by cell volumes. It calls no PoPS
operator/emitter/solver. A separate Fraction two-cell signed cross-flux proof
checks exact telescoping. Source discriminants reject reaction-times-T loads,
missing captures, mutated unknown additive sources, incomplete publication and
foreign partition indices; reversed prior carrier order remains admissible.

The seven Newton controls are imported unchanged from the captured diffusion
MMS: tolerance1e-10, max_iterations20, linear_tolerance1e-8,
linear_max_iterations240, restart60, armijo1e-4, minimum_step1/1024.
FD step1e-6 and acceptance3e-8 are declared before native execution. These
bounds are not tuned to a native result. Both steps must exercise nonzero
spatial flux. At step two there is no invented target: the oracle receives
saved Q1 and checks Q2-dt*(div(D(T2)grad(T2))+f)-Q1, Q2=Q(T2), the auxiliary
constraint and the amount balance independently.

The native test requires installed `pops` from the selected interpreter prefix,
uses the actual 2D native MPI world, collective call/check boundaries, a shared
collective directory and rank-zero compile publication. It captures each
actual Q/forcing State and physical T/z history independently. History slots,
fill, duration and typed sample identity are checked. Exact carrier image,
diagnostics, histories, lifecycle and numeric arrays are compared after
checkpoint reload and replay; final continuous/replay checkpoint bytes must
also match exactly. Forcing preservation is exact. Detached NPZ arrays are
marked read-only before scientific checks.

The receipt records actual artifact/platform identity, native DSO hash, all
block/program DSO paths and hashes, required real identity sidecars, actual
retained compiler command and compiler-owned dumped Program C++, initial NPZ,
accepted/reloaded/continuous/replay NPZ metrics, and checkpoint hashes. A
missing retained command/source/sidecar is a refusal, not synthesized evidence.
JUnit records artifact, dimension, rank, size and evidence path. ROOT's runner
must additionally retain its external source/SDK/native/launch identity pins.

## Source reception and pending execution

Source base: 5708718ef51dffcfb1ca43c06429cfd49d3d0b39. Production commits:
4b8932b8, 2289bd4d, 6528af09 (additive), 6d88b165 (partition),
41803be2 (additive capture endpoint authentication).
34 source/math tests pass (18 new, 16 historical Stage). Eight installed-native
cases collect successfully. Ruff and diff whitespace checks pass. No native
compile, JIT, installed environment mutation or MPI run was performed here.

Three fresh historical Stage profiles (scalar, reordered multicomponent with
auxiliary constraint, per-candidate diffusion) were generated at one stable
call site against base570 and final418. Full IR, CPP, Module hashes/manifests and
full SolveRequests are byte-identical, aggregate SHA256
4603a30de4c97b630a895664168563ca476efda0407e425ba4aa8b5efa7f0abe.

ROOT can execute its authenticated installed runner on
`tests/python/integration/runtime/test_public_evolved_original_stage.py`
(first serial, then MPI2), disabling pytest's source pythonpath (`-o pythonpath=`)
and unsetting PYTHONPATH. A new installed package containing these Python ports
is required; unchanged headers do not authenticate a newer Python body.
Source command (private checkout, never used as native evidence):

    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
      /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
      -c 'import sys;sys.path[:0]=["python",".","tests/review"];import pytest;raise SystemExit(pytest.main(["-q","tests/review/test_sol61_evolved_stage_mms.py","tests/review/test_sol61_evolved_original_field_stage.py"]))'

ROOT owns compilation, native reception, serial/MPI acceptance and evidence
pins. Native convergence, checkpoint bytes and collective execution remain
pending until that reception; these fixtures are not reported as positives.
