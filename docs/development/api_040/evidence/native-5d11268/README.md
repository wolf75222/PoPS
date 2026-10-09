# Native reception before the storage and boundary corrections

Actual installed package: SDK `242e849c6aadb2526d1ada0f54276c7db6be411de012f728a0817867fb155ea2`, 1,084 authenticated shipped files. Dim1 native `f2b66e8e06cf9f004e9bae7af151fe50482b7be5adbd3cf809e2ab56d11d03bf`; Dim2 native `73b3bfaa9a6c22380a3eb500e216b97c1978725a36476dd4f4a573dddf87a5a2`. Kokkos OpenMP CPU, MPICH; no GPU execution.

- The 19-test Dim2 receipt has nine passes and ten failures. All three coordinated-face tests pass against an independent asymmetric nonintegrable oracle. Eight purely local product/H05 cases fail before JIT because the emitter has no spatial storage carrier. Two legacy Path tests have an incorrect stability witness.
- The corrected Path fixtures pass all three selected native tests. The authored two-face frequency is 9.44, so dt=.1 is admissible under the unit FixedDt budget; dt=.2 must reject. Production thresholds are unchanged.
- Five true Dim1 HyQMOM B.1 tests pass: two short PDE oracles and three inadmissible-state refusals preserving state and clock. This does not qualify the full N32/64/128 campaign or MPI2.
- M07 fails at bind because admission requires the complete legacy boundary callback set even for a Path route. Import authentication succeeds; the runner cannot qualify the scientific backend because execution ends before its scientific receipt.
- C++ reception has 79 passes, one failed MPI aggregate and two single-rank skips (82 rows). The three-level serial AND9 rollback/retry passes. The MPI aggregate rejects at the intended refresh and preserves rollback/retry, but loses the original exception text in a generic wrapper. The diagnostic rerun records that exact failure. No MPI acceptance is inferred from this failed receipt.

Later integrated fixes require separate rebuilt receipts. These copies retain original commands, identities and XML assertions; raw source manifests, arrays and binaries remain in the workspace. `manifest.json` hashes copied execution files; `SHA256SUMS` covers this bundle except itself. Verify with `shasum -a 256 -c SHA256SUMS` from this directory.
