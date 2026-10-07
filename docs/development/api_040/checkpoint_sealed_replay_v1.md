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
