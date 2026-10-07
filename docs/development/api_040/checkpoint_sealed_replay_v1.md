# Checkpoint publication from a retained seal

`pops.checkpoint.sealed-replay@1` applies to publication retries of the built-in
`RestartV3` checkpoint provider. Its provider identity is
`pops.restart.accepted-state-v6`; a completed consumer publication identifies
`pops.restart-checkpoint.v6`. The constructor and checkpoint archive formats
(including AMR12), numerical equations and Native ABI12 are unchanged. The new
provider capability participates in the existing ConsumerGraph identity.

The motivating execution is an AMR outer step transaction. The checkpoint is
prepared and sealed while the candidate is authenticated. Publication happens
after commit, when the Native factory correctly refuses a new candidate capture.
Retrying by rolling back the failed publication and calling `snapshot` again
would therefore attempt an invalid second capture. This contract replays the
completed seal rather than reconstructing State, Field, history or clocks.

Before the first publication of a `Retry` checkpoint, rank zero acquires a
separate private transaction directory and an owned hardlink to the sealed NPZ.
The retained descriptor is read-only. A streaming SHA256 and exact length bind
the archive; its length must fit the live authenticated `max_archive_bytes`.
The original resealer's writable descriptor is closed. Peers receive only seal
metadata through the existing collective control plane, never archive bytes.

The optional `PreparedPublication.retry_publication` hook runs before terminal
rollback. For this provider it compensates the failed attempt and creates a
fresh private staging link to the same retained inode. Ownership transfers to
the replacement preparation only after staging agreement. It never calls the
Native factory, State gather or checkpoint resealer. The existing committed
candidate token is checked on every replay and publication: owner, session,
point, committed phase and lifetime remain mandatory. A stale token or peer
refusal prevents publication. Other providers return `None` from the hook and
keep their existing rollback-then-prepare retry behavior and attempt policy.
If a hook transfers ownership but its typed replacement fails effect/payload
validation, that replacement is compensated and its recoveries are retained.
The refusal does not fall back to a new capture of the committed candidate.

Only one archive inode is live; retention does not copy or double its physical
payload. The controlled link counts are one for the retained source alone, two
with staging or an accepted target, and three during target publication before
staging removal. There is no cross-filesystem copy fallback. Each digest reads
bounded chunks of at most 64KiB (and never beyond the archive budget). Retention
hashes before and after acquiring its link; a replay hashes before and after
staging; publication authenticates the seal before linking its target. These
extra reads have no measured performance claim.

Target collisions preserve the existing artifact. Attempt compensation uses
the existing descriptor-anchored quarantine and never deletes the retained
source. Discard, exhaustion, terminal rollback and finalization release that
source and its descriptors explicitly. A release failure remains owned for
the existing idempotent finalization retry. Cleanup errors preserve the first
failure and append release diagnostics. A lost control transport permits only
local rank-zero cleanup, with no further collective replay.
An exception from staging agreement also seals further replay, since agreement
itself may have lost its transport. Healthy root producer failures remain
retryable under the existing policy; these terminal control failures carry the
explicit `publication_retryable=False` marker.

The Source tests use real NPZ files and typed consumer effects solely as an I/O
unit fixture. They check one capture despite retry, exact file hashes, read-only
retention, collisions, exhausted attempts, archive budget, stale-token refusal,
peer staging refusal, discard and retryable finalization. They do not fabricate
a Native/PDE checkpoint or qualify dynamics. Native qualification must use a
new committed package and the public original IMEX publication failure/retry
witness with its unchanged scientific oracles.
The prepared Native test is
`test_imex_checkpoint_sealed_retry.py::test_original_imex_checkpoint_retry_replays_one_completed_seal`.
It assembles the original public authoring helpers and changes only the resolved
checkpoint manifest to `Retry(2)`. It compares a reference run against the
publication-retried run, retains both genuine publication files, counts the
candidate factory and seal, restarts from the replayed checkpoint, and applies
the original exact State/Field/Program/regrid and continuation oracles. It has
not been executed by Source validation.

## Original Native retry received, 7 October 2026

The unchanged original witness now passes on the installed production package at
`ede680fe00746c1bc7e5eeb4afabd9772b92cc33`, Native13/System package8 and checkpoint
provider v6. The installed Dim2 extension is `ea0c7bf4e2a752469091030597cedd671291fc74bd9354ab8fbcdb6068eeec1d`,
SDK signature `4b2c57fbdc84b72f8b0c66c7c943eb2f793d519181d7f9032f02313032e8f04a`.
The pytest process records actual Host OpenMP concurrency2 and MPI rank0 of1.

One candidate factory produces one captured seal. Both publications reuse its
178807 bytes and device/inode pair `[16777231,338240147]`; the two retained files
and published target are byte-identical, SHA256
`b5acae1632f72b5032cc914f051b179b40d2c233d574afc1af2f828ac538d0d2`.
Reference, accepted and restored State/Field arrays, Program data, full snapshot JSON
and consumer cursors are exact; continuous and restarted trajectories are exact.
The checkpoint has time1e-4, macro step1, two levels and one rank. The continuation
changes patch boxes and data while retaining two levels: regrid counters and topology
epochs change from2 to3. No growth to three levels is received.

The saved flux ledger has920 entries at levels0/1; three clock entries, two
synchronization entries, diagnostics and temporal state are populated. Histories
and cache are empty lists in this witness. It therefore does not receive populated
history/cache retry semantics. An independent offline reader verifies69 actual
artifacts, the original fixture/oracles and the installation's2793 unchanged entries.
ROOT positive receipt `root-own-checkpoint-retry-native13-positive-20261007.json`
has SHA256 `1031e8befc07b2b38c0e4d5a147d2b46d0ecd408ab0e244eab1299e5809640e9`.
This is one original CPU Dim2/MPI-enabled world1 case; MPI2, GPU, Dim3, concurrency
between simulations and all94 mission obligations require their separate receipts.

From the initialized `pops` environment and integrated checkout, reproduce with
the official installed-package driver and a fresh output directory:

```sh
rtk proxy env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 POPS_INCLUDE=/Users/romaindespoulain/dev/tmp/PoPS-api040-integrated-main-20261004/include Kokkos_ROOT=/Users/romaindespoulain/miniforge3/envs/pops POPS_KOKKOS_ROOT=/Users/romaindespoulain/miniforge3/envs/pops CMAKE_PREFIX_PATH=/Users/romaindespoulain/miniforge3/envs/pops POPS_NATIVE_DIM=2 POPS_REQUIRE_NATIVE_TESTS=1 OMP_NUM_THREADS=2 OMP_PROC_BIND=false /Users/romaindespoulain/miniforge3/envs/pops/bin/python docs/development/api_040/run_installed_checks.py --output /tmp/pops-api040-original-checkpoint-retry --test tests/python/integration/runtime/test_imex_checkpoint_sealed_retry.py
```
