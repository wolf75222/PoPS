# Declared quadrature domain M18 / W09

Library contract `pops.discrete-entropy-certificate@1` is authoring algebra.
`DiscreteEntropyCertificate` accepts an exact declared quadrature and arbitrary
basis-order covector, checks exact rational signs of its binary coefficients at
**every** retained basis column, and requires a positive column margin. It makes
no assumption about density slot, monomial order, component names or model.
It does not claim to enumerate all facets. Its immutable descriptor is explicit.

The script fixes positive weights `(1/3,1/3,1/3)`, nodes `(-.5,0,.5)` and basis
`(1,v)`. Original W09 supplies the interval and target, not a unique weight
table. This table is declared throughout all three cases; the historical M18
five-node/three-moment example is byte-preserved.

For y=(.5,-1), the target (1,.9) has margin -.4 and is certified outside the
cone. Target (1,.5) has zero margin: nonnegative endpoint populations can realize
it, but positive finite exponential populations cannot. Separate native guards
name these two refusals. Target (1,.49) is inside; the common Newton header
converges at residual 4.04e-14 without changing the prescribed target. Guards
are authored by public `value/min/guard`, before `LocalResidual`, not new
compiler/runtime opcodes. Numeric FP acceptance thresholds are unchanged.

Prospective installed test:
`tests/python/integration/runtime/test_m18_w09_declared_domain_runtime.py`
has three parameterized nodes, suitable for ROOT CPU/MPI2. It requires C25 ALL
models, preserves actual Model/Program source and companion binaries before
bind, then saves local/complete storage before valid getters, Target/dual arrays,
public cursor metadata and clocks, attempt failures and checkpoint. Refusal must
retain its exact diagnostic and bit-equal full accepted State/clock/cursors;
positive must reconstruct original moments within 2e-11, finite positive
populations and a strictly higher-entropy moment-nullspace perturbation.
The closure has no declared history variable; cursor/checkpoint metadata is
retained, not an invented State history. No Native execution is claimed here.

Source/host commands use `env -u PYTHONPATH`, existing ir17 and absolute
`pythonpath=<WT>/python <WT>` with `--noconftest -p no:cacheprovider` for:

- tests/review/test_sol61_m18_w09_declared_certificate.py
- tests/review/test_sol61_m18_w09_header_host.py
- tests/review/test_sol61_m18_independent_review.py
- tests/review/test_sol61_m18_entropy_offline_contract.py

Original coherent run: 81 PASS / 1 FAIL, preserved XML externally. The sole
failure is the already-present current-version consumer asserting Native ABI5
against ABI8 in exact parent717. Parent proof uses its actual test function and
all three exact Git file inputs; it is not a Native failure. A separate consumer
migration commit ties this current test to generated release authority, not to
any historical archive reader. Five frozen readers remain unchanged.

Official setup executed once with `POPS_ENV_NAME=pops-sol61-m18-w09
POPS_NATIVE_DIM=2 bash scripts/setup_env.sh --cpu --dim 2`, exit0. It created the
isolated ENV and explicitly reported PoPS not installed; no Native build/run.
132 existing pops/ir17 conda-meta files were hash-equal before/after. Setup and
Source receipts live externally under `sol61-m18-w09-original-preparation`.

Root must integrate/build and receive actual scientific/refusal archives. No
arbitrarily-near-boundary robustness, full M_N/article algorithm, transport,
MPI/GPU, performance, CI or complete mission qualification follows from Source.

Certificate @2 normalizes by an exact power of two, checking every binary coefficient and every node again. Unrepresentable dynamic range is refused at authoring. Runtime products/sums carry a conservative binary64 forward-error envelope: values within it are `finite_dual_not_certified`, never a numerical proof of a boundary. Nonfinite arithmetic is explicitly indeterminate. The declared W09 boundary has a separate mathematical proof; no target is repaired.
