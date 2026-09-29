# M01/W01 independent review of saved native states

Date: 2026-09-29. Reviewer: GPT-6 Sol, independent of M01 authoring. Evidence:
`/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/outputs/m01-openmp1-reviewed/result.json`
and its four `state_N.npz` files. Current checkout: branch
`codex/api-040-native-20260928`, HEAD `022f5acbb9181eb85a6d9cd05e30a92aad1c4872`;
the M01 script remains untracked in this shared checkout.

## Independent recomputation

I reopened all four state files and recomputed the analytic mean directly from
edge integrals,
`1 + 0.1 N [sin(2π(x_{j+1}-t))-sin(2π(x_j-t))]/(2π)`, without calling the
example's `exact_cell_averages`. Native arrays have shape `(1,N,N)` with x
last. At all N, saved exact values agree with the independent edge integral
within `4.0e-15` and bound initial values within `5.2e-15`. The initial cell
mean differs from a center sample by `1.02e-4` at N=40 down to `1.61e-6` at
N=320; the test did not merely compare point values to point values.

| N×N | Recomputed L1 | Mass defect | max y variation | Steps |
| --- | ---: | ---: | ---: | ---: |
| 40×40 | 2.997736458275449e-4 | 0 | 1.11e-15 | 17 |
| 80×80 | 7.112465922364211e-5 | 0 | 1.33e-15 | 34 |
| 160×160 | 1.689517328649047e-5 | 0 | 2.89e-15 | 67 |
| 320×320 | 4.052047833988463e-6 | 0 | 4.22e-15 | 134 |

Independent log2 error ratios are `2.07545, 2.07374, 2.05989`, all above
the predeclared 1.6 threshold. Recomputed L1 differs from the receipt by at
most `3.7e-17`. The x variation remains about `0.17`, so y invariance is not
an accidental spatially constant solution. All values are finite; final
times equal `0.125`; minima and maxima stay inside the predeclared
`[0.9-1e-12,1.1+1e-12]` range. All four run reports show the target-time stop,
zero rejected steps, and distinct run/artifact identities. I independently
reapplied every recorded acceptance check; all four pass.

## Artifact and scope checks

Every `state_N.npz` SHA-256 equals the corresponding receipt hash. The receipt's
package path equals the currently imported installed `pops.__file__`, and its
native module path, SHA-256
`e1070cb63a8e733b0a6032ac4d2b8ad003756668c100e057e5f8edd5b9c01280`,
and `abi_key()` equal the currently loaded installed Dim=2 module. Backend
evidence says OpenMP CPU, float64, MPI world; the four runs used one MPI rank
and requested one thread. This is a native Dim=2 N×N result for a y-invariant
1D physical solution. It does not qualify a Dim=1 build, multiple MPI ranks,
or another execution backend.

The receipt authenticates state files and the installed native bytes. It does
not contain the untracked M01 script's source hash or a source commit; it cannot
by itself prove which exact script bytes produced the files. The current
script's equation, exact-average formula, acceptance checks, and saved-state
flow match the recomputed evidence. No M01 oracle/test/criterion change was
needed or made.
