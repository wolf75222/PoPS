# Review artifacts outside the code tree

At `b6e64c1d` (PR #687) the code tree carried the review material of the API 0.4.0 and
API 0.2 work: review reports, reception receipts and run logs, evidence bundles with
captured source snapshots, review-only probes and tests that CI never collected, and the
core-audit evidence of #686. That material left the tree in one change (Linear ROM-84, N5;
spec 0.3, rule MG5). This page says where it is and how to check that nothing was lost.

| Directory at `b6e64c1d` | Git tree | Files | Bytes | Content |
|---|---|---:|---:|---|
| `docs/development/api_040/` | `4f84f0b7` | 2192 | 41605602 | API 0.4.0 review reports, receipts, evidence bundles, logs |
| `docs/review/` | `da31f8c3` | 466 | 10289854 | review notes and the core audit of 2026-10-12 (#686) |
| `tests/review/` | `f577ca5c` | 548 | 3961327 | review probes and tests outside `testpaths` |
| `outputs/` | `047a2498` | 62 | 221112 | review notes and run logs |
| `docs/dev/` | `9aeb3e44` | 13 | 61100 | change notes written for independent review |

## Where the files are

- In git: every file stays reachable as `git show b6e64c1d:<path>` (commit
  `b6e64c1d23f5c0565c41674d262ca26b401e8339`, head of PR #687 and of
  `refs/pull/687/head`). `git rev-parse b6e64c1d:<directory>` returns the tree ids above.
- Outside the repository: `pops-review-artifacts-b6e64c1d.tar.xz` (3747660 bytes, SHA-256
  `6b09f43c0ab1ed2d550d66327fbf70331f29b618ee03b7817b017ce6fd2e8969`) holds the five
  directories byte for byte, written by `git archive` from `b6e64c1d`. Its `MANIFEST.tsv`
  gives, for each file, the path, git blob id, size, SHA-256 and where the file went;
  `FILES.sha256` repeats the hashes in `sha256sum` format. The archive and both lists are
  attached to Linear ROM-84 and kept in the PoPS project files under `landing/archive/`.

Check an extracted copy with:

```sh
sha256sum pops-review-artifacts-b6e64c1d.tar.xz   # the hash above
tar -xJf pops-review-artifacts-b6e64c1d.tar.xz
cd pops-review-artifacts-b6e64c1d && sha256sum -c ../FILES.sha256
```

## Files that stayed in the tree

The remaining code, tests and docs read 17 of these files and two receipt bundles, and seven
benchmark notes describe scripts that stay. They moved instead of leaving, with the same bytes
unless the last column says otherwise; the archive keeps their original copies too.

| Path at `b6e64c1d` | Path now | Change |
|---|---|---|
| `tests/review/sol61_amr_full_carrier_offline.py` | `tests/python/support/` | none |
| `tests/review/sol61_tag_selection_oracle.py` | `tests/python/support/` | none |
| `tests/review/sol61_m19_freestreaming_oracle.py` | `tests/python/support/` | none |
| `tests/review/sol61_m19_amr_consumed_offline.py` | `tests/python/support/` | its import of the carrier decoder |
| `tests/review/test_sol61_mapped_field_route.py` | `tests/python/support/sol61_mapped_field_route.py` | renamed so pytest does not collect it; repository root found one level higher |
| `tests/review/sol61_t5_public_native_capture.inc` | `tests/cpp/integration/runtime/` | none |
| `docs/development/api_040/sol61_global_authority_parent_parity.json` | `tests/python/support/fixtures/` | none |
| `docs/development/api_040/check_finite_amr_converged.py`, `check_m23_m05_converged.py` | `tests/python/support/fixtures/receipt_bundles/` | none |
| `docs/development/api_040/evidence/finite-amr-converged/`, `m23-m05-converged/` (327 files) | `tests/python/support/fixtures/receipt_bundles/evidence/` | none |
| `docs/development/api_040/joint_reconstruction_*.py` (7 files) | `benchmarks/joint_reconstruction/` | none (the resource probes and their tests pin the benchmark scripts by SHA-256) |
| `docs/development/api_040/joint_reconstruction_*.md` (the 7 notes of those scripts) | `benchmarks/joint_reconstruction/` | one link in the v1.1 note, one in the v1.2 note |
| `docs/development/api_040/block_inverse_scaling_v1.md` | `docs/development/` | none |

The tests that import or read them now name the new paths. Four documents that cited an
archived file by path now say that it is in the archive.
