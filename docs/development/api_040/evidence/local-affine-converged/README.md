# Frozen local-product and affine reception

The XML and raw byte hashes were checked independently from the runner summaries.
No experiment was rerun to create this archive. `inventory.json` preserves every
test node and its module; `manifest.json` and `SHA256SUMS` authenticate the copied
payloads. The build log is `build-selected-state-waves-dim2.log`.

| Receipt | Exact acceptance | Duration |
| --- | --- | --- |
| Serial converged fixture suite | 26/26: 6 helper units plus 20 installed native tests | 418.896079625 s |
| MPI2 suite | 23/23 on each rank, the same 23 cases; not 46 independent experiments | 505.306192666 s |
| Selected-state/waves units | 28/28 against the installed package | 6.868926291 s |

The 20 common serial/MPI native cases comprise local products (4), H05 (2), H05
independent adversaries (2), source/apply products v2 (2), read-only capture (2),
affine library bodies on stages/unknowns (4), conditional diffusion/old affine/
affine library (3), and particle-oracle old/new affine equivalence (1). MPI adds
the two M11/W10 constrained-product permutations and one M18 closure test covering
twenty targets and repeated outside-cone refusal. Their separate serial evidence
is retained in the older `post-7b35ffd-scoped` bundle; these three tests were not
part of the new 26-case serial run.

All runs select installed **Dim2, CPU Kokkos**, MPI-enabled native library, one
thread per process. The MPI run uses two processes. No AMR, GPU, other native
dimension or complete physical corpus is qualified by this bundle.

- DSO SHA256: `00b38fe09ba42678a5a2beda64a3767be9aa130455ecdfcf864a7182fc519cc9`
- SDK/header signature: `02723ae9a5d36640fb5ad31d3e89c9b7b3a4a097c92059aac57d2857a3b4020e`
- 1,086 source files; manifest SHA256: `884ce7a7893c3bab92c04e2dcf940242bdb1b6baf1ef948b640d837ba3c20bb0`
- Serial and MPI-before source commit: `941afec2404299bd7945ba6eff31642f5b71690a`
- MPI-after commit: `a16b2ccaa69e3b0c017fec2aec9d4eb72facd9d6`
- Installed-unit commit: `34faea0fd3e984673a616f61f93552dc06f7f198`

The MPI commit change covers only documentation and archived evidence. Before
and after production manifests are byte-identical; both ranks authenticate the
same DSO, selected dimension and package location. At archival time every listed
source hash was also compared with the isolated audit checkout at `3b52036a`,
and the live DSO bytes matched the recorded hash. Eleven runner/test source files
match the MPI runner manifest and are copied into `source-snapshot`. The helper
and its unit test are copied additionally from the clean audit base; the runner
did not separately list these two files in its test-source manifest.

Historical failures remain in earlier bundles: missing storage authority,
carrier/IR/compiler defects, the M18 rectangular fixture, and two old conditional
fixture failures caused by comparing an exception's exact name rather than its
RuntimeError ancestry. Passing the present matrix does not rewrite those runs.
The native criteria and equations are unchanged. This archive contains test
receipts, not a new collection of persistent scientific NPZ trajectories.

Run `python docs/development/api_040/check_local_affine_evidence.py` or the full
`check_registry.py` for read-only verification. No PoPS import or build occurs.
