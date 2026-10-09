# Independent vector pairing witnesses before dot_all implementation reception

Date: 30 September 2026. Reviewed source: receiver ff07a7ce; the production
`dot_all` extension is not included in this reception.

Actual Uniform and AMR `dot`, `norm2`, and `norm_inf` methods explicitly reduce
component zero. The independent test extracts those six complete methods unchanged
from their source headers and compiles a small host harness. Eleven checks receive
their component selection, local-error convergence before the numeric collective,
missing AMR right-field refusal, empty AMR participation, and inactive poisoned cells.
Storage, active masks, field-contract validation and serial collective functions are
explicit substitutes. This proves the extracted method control flow, not native
MultiFab ownership or MPI behavior. A nonfinite tail is intentionally invisible to
legacy dot; this historical behavior is retained, not endorsed as a vector norm.

Thirteen tests in `tests/review/test_sol61_vector_reduction_oracles.py` pass in
1.07 s, including that host probe and public authoring IR/refusals. They provide
independent discrimination matrices for the new capability:

| Witness | Expected distinction |
| --- | --- |
| q=(2,3,5,7,11), rotate first two coordinates by pi/2 | All-coordinate self-pairing stays 208; legacy component-zero value changes 4 to 9. |
| Multiply q's tail by ten | Legacy component-zero self-pairing stays 4; vector value becomes 20404. |
| q=(0,3,4) | Component-zero self-pairing is zero; all-coordinate value is 25. |
| 5D skew matrix, planes (0,2) frequency .7 and (1,4) frequency 1.1 | Exact cosine/sine oracle preserves 208 at four times and under permutation (4,2,0,3,1). |
| Gram diag(1,2,7), transformed skew evolution | Declared weighted pairing stays invariant; raw Euclidean coordinate pairing changes. |
| Coarse/fine owned masks, inactive NaNs and covered coarse cell | Raw vector pairing 104; component-zero 39; volume-weighted physical value 29.25. |
| Widths 1,2,3,5 and signed pairings | Joint permutation preserves pairing; scalar case agrees with historical component zero. |

The mathematical API0.4 authority is
`PoPS_Codex_handoff_0.4.0/reference/PoPS_API_v0.4.0/sources/mathematical_original.tex`,
lines 1033–1053 (trial/test/dual roles and declared positive Gram matrix) and
1075–1084 (heterogeneous units, component scales, weights and measures).
These are read from the authority checkout, not inferred from test tolerances.
A raw all-coordinate contraction is appropriate for explicitly compatible coordinates;
it does not supply an automatic physical norm for mixed density/momentum/energy units.
The W05/W06 rotation witnesses in the API specification do not authorize changing
historical `dot` semantics. No diffusion coefficient or PDE qualification is inferred.

The minimal compatible new mechanism is an explicitly versioned `dot_all` reduction.
Old names, attrs and hashes must remain unchanged. The new path must preserve owner,
block, layout and component order, skip AMR covered/inactive cells, and perform local
kernel-error convergence before one sum on the prepared execution lane. Existing
global `pops::dot_all` helpers cannot substitute for this control flow: some perform
their own communicator collective, and some use physical measure weighting.
An explicitly requested Euclidean norm can be expressed as `sqrt(dot_all(u,u))`.
Metric/scales/measure choices remain declared mathematical data.

Reproduction:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/review/test_sol61_vector_reduction_oracles.py
```

This commit supplies oracles and historical compatibility evidence. Native Uniform/
AMR collective pairing, MPI/GPU, rotation integration and any new serialized IR
version need reception on the future exact candidate; no new capability is marked
received by these tests.
