# Independent finite-sharing fix reception - 30 September 2026

Author fix `1c84af06bad8f86680427e1066d8ec64481ad68c` was applied alone on the
exclusive review tree containing `908508a`, yielding
`7eecb6de24c628e03a0697a503bc5c09586abdde`. Unrelated diagnostic commits
`074…`/`84c0e73` were not applied. The relevant production/test blobs are checked
byte-identical to the author's fix. Principal checkout and shared environment
are unchanged.

**The mixed-expression defect is closed. A finite-vector arithmetic identity
regression remains.** This is source authoring/hash reception, not native
numerical reception.

The exact-parent comparison replays seven mathematical DAGs, three policy
identities, handle inspections, real Program authoring, ten refused conditions
and unchanged layer gate. Mixed scalar/finite projection encoding now contains
one application, and its complete Program image/hash matches parent
`bc37b0af4012a10fa3e9bd7ecaccd1f7befc897b`. Plain finite materialization also
matches. The comparison receives 32 checks. Refusal wording/type differences
recorded at the `908508a` review remain explicit, without any accepted invalid
condition.

## Remaining Program-hash difference

`(vec + vec * Fraction(2, 3)).materialize(program, ..., template=u.n, at=u.next.point)`
has the same structural mathematical DAG, but encoded constant sharing differs.
The parent encoded eleven nodes; the receiver encodes ten for the independent
literal-vector probe. The real authored Program hashes are:

- Parent: `c7f6fdecbb652608c4fe67afeb38ed9bda28c72a7d4fd10ee2dba1ac7205aed1`.
- Receiver: `d60156c68eb4ba8c06fa54a8f47a3a5e282d9800fdece23f0832483af6f2ff45`.

The original finite-vector multiplication called `_wrap(scalar)` separately per
component, producing separate constant objects for an ordinary numeric scalar.
The replacement retains that scalar once, then shares its scalar plan between
components. The weak lowering cache fixes repeated projection conversion but
cannot restore these originally distinct scalar declarations. This is the
remaining `908508a` regression, not a new numerically wrong trajectory claim.
The integrating worker and author were informed. Preserve the historical
encoded sharing or explicitly version any intended semantic identity change;
do not report all Program hashes unchanged yet.

## Independent weak-cache/refusal probes

Nine added tests in `tests/review/test_sol61_finite_weak_cache.py` verify:
repeated mixed conversions share the same projection/application while alive;
an explicit exact `Const` reused in both inputs stays shared; vector self-addition
retains reused scalar identity; releasing the IR expression permits collection
while the immutable declaration can still be reified; 100 retired declarations/
applications/expressions leave no owned objects or cache keys; malformed version,
operation, arity and application are refused before cache insertion; cycles,
mutation of frozen plans and Python truth remain refused.

The cache owns only weak references to declarations and results. Its source
identity check protects against stale Python object IDs, and the release callback
only removes the row holding its own weak source reference. A declaration can
outlive its result without retaining compiled IR. Foreign declarations lacking
weak-reference support share within a batch rather than gaining cross-batch
identity guarantees. No import direction or architecture allowlist changed.

138 selected source tests pass in 12.90 s, including the nine independent tests,
three new author tests and the earlier 126-test selection. No test skips, native
build, native JIT, package install, environment edit, MPI or GPU execution are
claimed. Source tests explicitly unset `POPS_NATIVE_DIM`; the isolated tree has
no native variants manifest. Ruff and whitespace checks pass.

```sh
rtk proxy /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  tests/review/sol61_layering_1c84af0.py
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q \
  tests/review/test_sol61_finite_weak_cache.py \
  tests/python/unit/numerics/test_symbolic_policy_layering.py
```

The complete 138-test selection is the earlier `908508a` review command with the
weak-cache test file added. Raw receipts live under
`outputs/sol61-layering-1c84af0/receipt.json`; source archive SHA-256 is
`23098e5071a34aeaf670df44097264d10247f4c3281370994f5854879f4bc2ce`,
parent archive remains
`b2a8d77345da0cf03f0f297c3543ac8f7838c60686e101a7d5e65b0f10a53d53`.
The script prints the received mixed fix separately from the confirmed residual.
