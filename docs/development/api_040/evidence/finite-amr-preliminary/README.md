# Preliminary finite-support and AMR receipts

This immutable historical bundle preserves a failed installed reception and a
bounded fixture repair. It does not establish M09/W06 runtime completion.
Registry conclusions are deferred until the repaired native cases are received.
Later successful runs must be added separately, without overwriting these files.

| Receipt | XML outcome | Duration | Source commit |
| --- | --- | --- | --- |
| `installed-finite-amr-units` | 118 passed, 1 failed / 119 | 103.462 s | `70c4724835cfad79cd10d0aed9ae1a8278304f3d` |
| `installed-amr-flat-authority-repaired` | 13 passed / 13 | 5.195 s | `954b7dcf08d0642db4f3556cafbf7f994b5374b1` |
| `installed-finite-amr-native-serial` | 2 passed, 7 failed / 9 | 173.831 s | `70c4724835cfad79cd10d0aed9ae1a8278304f3d` |

The 13-case repaired file overlaps the 119-case suite; these counts must not be
summed as independent tests. The single unit failure is a diagnostic-regex
mismatch in `test_flat_plan_still_requires_a_matching_resolved_state_method`;
the actual refusal is retained in XML. The native failures comprise six JIT
compilation failures (`expression_active_*` referenced without declaration,
followed by mask argument conversion errors) and the two-level affine fixture's
failure to create a partial refined patch (`256 < 256` fails). A one-level
affine run and the impossible-local-residual rejection pass. The six generated
`.failed.cpp` files are copied byte for byte, with their original pytest/cache
directory names. No failed attempt is reclassified as numerical acceptance.

All three receipts authenticate the same CPU Kokkos, MPI-enabled **Dim2** native
extension (the campaigns here are serial):

- DSO SHA256: `aed8c1582c3344bd5afb19ab784adec260bcdbd4bf944b26a29242b53889c862`.
- SDK signature: `396719235acdb96435dba0dd27c5e5579aefbda4936f657262b9b2553d13bb4c`.
- Source manifest SHA256: `6c432f8e7005a26400f852e24d3f11b9443eeac0d79cac2985fc549f869c13fc`.
- 1,089 production file hashes. The reviewer checked every hash against the
  corresponding Git blob at each of the two recorded commits; both sets match.

The source commits differ, but the production source manifests are identical.
Test snapshots were recovered from each runner's clean recorded commit, as
specified in the payload manifest. These are distinct from an individually
authenticated test-source manifest, which this older runner did not produce.
No claim relies on whichever package happens to be installed after the run.

The three build logs preserve the real Dim1 build, Dim2 build, and the later
Python-only materialization repair (`ninja: no work to do`). Both native builds
finish with their dimension-specific doctor checks. A Dim1 build log is not a
Dim1 scientific receipt. The Python repair log likewise does not turn the
preceding failures into passes. Repaired unit/native receipts are intentionally
outside this preliminary archive.

`inventory.json` is derived directly from XML and reconciled with runner
`result.json`. `manifest.json` records every payload's origin, size and SHA256;
`SHA256SUMS` also authenticates the manifest. Run the dependency-free verifier:

```text
python docs/development/api_040/check_finite_amr_preliminary.py
```

No tests, build, JIT or native import were launched to prepare this archive.
It provides no MPI execution, GPU, general AMR, or full corpus qualification.
