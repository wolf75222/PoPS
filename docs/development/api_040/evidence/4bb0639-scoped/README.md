# Historical `4bb0639` execution receipts

This bundle copies 38 execution files byte-for-byte from the workspace outputs
directory. `manifest.json` records each original relative path, size and
SHA-256. `SHA256SUMS` covers every copied file plus this README and manifest;
it excludes itself. The source commit recorded by all three runner identities
is `4bb0639e33598b6c0aa0d1198ba74e5b5d5200db`. The installed Dim1 native
hash is `84b7d444…`, Dim2 `7105935a…`, and both report SDK headers
`02723ae9…`. These receipts precede the periodic MPI and qualified-parameter
repairs; later source or binary results cannot turn a historical failure green.

`installed-4bb0639-products-operators-frontier` failed: its XML has 21 tests,
11 passes and 10 failures. Eight failures report missing exact ghost depth for
purely local product/H05 blocks; two reject distinct block-owned `gain`
parameters as conflicting metadata. The passes cover eight selected
SymbolicPath protocol checks, two C17 original-residual cases, and one native
ExternalTimeGrid frontier case. They do not prove the failed local-product/H05
cases or a general C22 reached-time contract.

`installed-4bb0639-dim1-m10-mpi2` passed its two selected tests on each of two real
MPI ranks: each XML contains one source/emission check and one installed native
N32 one-step self-consistent Poisson-stage/joint Scharfetter–Gummel flux check.
The runner reports before/after authentication success, unchanged test sources,
rank parity and the exact Dim1 native hash. This is not a multi-grid or long-time
M10 campaign.

`m15-4bb0639-dim1-mpi2` failed after a diagnosed collective mismatch and
intentional interruption. It saved only `state_32_01234.npz` from the first
canonical trajectory; it produced **no scientific receipt**. The recorded
samples show rank 0 in `WorldCommunicator::allgather_bytes` while rank 1 was
still in `System<1>::step`'s collective rejection phase. The runner's
`returncode=15`, `correct_backend=false` and `scientific_receipt_sha256=null`
are retained exactly. This is a failure witness, not a six-run M15 MPI result;
the earlier serial Dim1 M15 receipt remains separate.

The two qualified-parameter logs are bounded source/host diagnostics on the
installed pre-fix inspector. The first stops at a test-fixture `BlockRegistry`
lookup error. The repaired fixture reaches the real inspector and produces two
failures for homonymous block-owned `gain` metadata. Neither log is a native PDE
test or evidence that a subsequent repair passed.

Verify from this directory with `shasum -a 256 -c SHA256SUMS`, then run
`python3 ../../check_registry.py`. The bundle captures historical evidence only.
