# Independent source/host review: Exp and finite-quadrature M18

Reviewed author commits `d846e5a` and `4619a34` against base `4bb0639` in an isolated
checkout (review cherry-pick aliases `38a58d6`, `a1f0413`). No production change was
needed by this review. The installed package has not been received by the reviewer.

The `exp` extension has a distinct structural/CSE key and closed Program DAG
opcode. It participates in dependency traversal, quantity substitution, physical
source emission, bind-expression reference discovery and evaluation, and the
common checked expression emitter. Its derivative is the chain rule
`exp(a) * da`. Generated source and mathematical expression keys change with the
new node; there is no collision with a previous arithmetic node. Keeping the
existing containing IR schema for an additive opcode is coherent here: older
emitters reject unsupported opcodes, and no native package ABI layout changes.
This does not waive rebuilt Python/native/SDK artifact authentication.

The bounded legacy `CoupledSource` bytecode and `fields._program_expression`
load-expression language do not accept Exp. They explicitly refuse an unsupported
node; this review does not qualify an exponential on those separate realization
routes. The shared Expr/physical source/pointwise/LocalResidual routes are the
ones exercised here. Positive mathematical exp may underflow to binary64 zero;
this primitive is not an entropy feasibility classifier or a positivity floor.

## Independent numerical and identity checks

The added `test_exp_entropy_independent_review.py` supplies six checks:

1. A nontrivial chain derivative and the complete public M18 Program's unchanged
   hash/C++ emission after detachment.
2. Seven nodes and four permuted/mixed basis functions, comparing populations and
   moments to independent NumPy matrix algebra; the log-density gradient annihilates
   the moment nullspace, and a positive feasible perturbation increases entropy.
3. Two instances of one model with the same parameter name retain distinct qualified
   expression keys and distinct bind values. A foreign-only environment is refused.
4. The three-moment realizability cone is checked independently as the polygon
   through `(v,v²)` after normalization by mass. Its piecewise-linear lower edges
   and upper edge agree with the author's LP oracle on all twenty targets and
   additional interior, edge and outside witnesses.
5. Simulated allgather delivery of a peer's assertion failure makes every caller
   refuse before a following collective. This is a helper test, not MPI execution.
6. A small strict host C++ compilation of the real common emitter shows that
   `minimum(exp(x),1)` cannot mask either exp overflow or a nonfinite input leaf,
   while an inactive exponential branch remains unevaluated.

Combined with the author's Exp, entropy-oracle and emitted-residual host files:
**17 tests passed in 4.20 seconds**, under `env -u PYTHONPATH`, source imports from
the isolated checkout, with no JIT, native package build or installation. Compiler
probes use ordinary host C++ only. The commands were:

```sh
python -m pytest -q --tb=short -o pythonpath=python \
  tests/python/unit/moments/test_exp_entropy_independent_review.py \
  tests/python/unit/codegen/test_symbolic_exp_local_residual.py \
  tests/python/unit/moments/test_m18_discrete_entropy.py \
  tests/python/unit/moments/test_m18_entropy_host.py
```

## MPI fixture correction and outstanding reception

The original installed fixture asserted rank-local accepted clocks and rollback
clocks/envelopes before a later collective `_root_check`. A divergence in the
invariant being tested could make one rank exit and leave another in broadcast.
Review commit `ed7fed8` makes these local checks converge through the existing
all-rank error-gather helper before asserting. No tolerance, state equality or
rollback condition was weakened. The author incorporated this fix as `bfb22b5`.

The original six-unknown product contains three identity equations for the target
state. That is not necessary mathematically; it worked around a read-only capture
being omitted from artifact bind inputs. Root is repairing that separate generic
input-discovery route. Author follow-up `ab5ecbd` was independently reread and
rerun: **17/17 source/host tests passed in 3.80 seconds**. It retains three dual
unknowns with the external immutable target capture, one dual commit, and the
unchanged original residual. The emitted host solve now has width three and checks
its reported original-residual norm. The MPI-converged fixture assertions remain.
No target identity equation or target publication is used to force bind discovery.
The review alias `ef15715` need not be integrated alongside the author's SHA.

The handoff distinguishes infeasibility on the retained quadrature from a feasible
boundary target with no finite dual multiplier. The pure oracle implements that
distinction; the native solver still reports a generic numerical failure. Moderate
interior solves plus one impossible second moment do **not** qualify near-boundary
optimization, dynamic native `E-CLOSURE-INFEASIBLE` versus `BOUNDARY`, arbitrary
quadrature conditioning, transport, AMR or GPU. The original W09 example with nodes
in `[-0.5,0.5]` and requested mean `0.9` remains distinct from this five-node
`[-1,1]` variant. Rebuilt installed reception, including the MPI rejection/rollback
fixture, remains mandatory before any native success claim.

## Current saved-state reception addendum (source base 48871851)

The historical source/host counts and receipts above are preserved. The expanded
fixture and independent offline contract are described in
`m18_saved_state_offline_reception_sol61.md`. They retain three multipliers,
the exact immutable target, one dual publication and the original 12-iteration
Newton budget with tolerance `2e-11`. They add actual native checkpoint/support
capture, two rejected attempts, an explicitly fresh safe rebind and owner-sealed
offline reception. Native execution of this expanded fixture is pending.

Current source has native module capability ABI **5**, separate System package
ABI **7**, and local M18 Program IR version **5**. The early generated-package
ABI/schema wording does not authenticate a contemporary installed package.
