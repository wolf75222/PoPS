# Accepted AMR trace coverage at the finest owner

The installed Dim2 AMR two-level witness at
`outputs/installed-integral-native-author-serial` rejected after a real native
step: `q=0.724`, whereas the authored initial value and right-face current give
`0.7 + 1.2*0.01 = 0.712`. The independent retained ledger in
`outputs/integral-amr-duration-probe` discriminates the cause. The two fine
substeps each have 64 physical right faces at duration `0.005` and amount
`-0.006`. Their combined amount is the required `-0.012`. The same ledger
contains 32 coarse right faces at duration `0.01` and amount `-0.012`, although
the public composite coverage mask marks every coarse right-boundary cell as
covered. The failure is **IMPL**: the accepted-exchange producer consulted only
the embedded-boundary activity mask, which is absent on this ordinary mesh.
The temporal weights themselves were correct and have not been changed.

The native context now exposes the prepared finest-owner coverage separately
from embedded-boundary activity. The Uniform context returns no coverage
restriction. Accepted transport and diffusive face producers require both
masks, when present, before staging a cell's incidences. This uses the existing
`prepared_amr_block_level_coverage_mask` authority and its exact block, layout,
distribution and rank checks; it does not change source equations, subcycling,
reflux, or the persistent scalar's consume/rollback transaction. A covered
coarse exterior face cannot be counted again with its fine descendants.

The source emission test checks both masks and the exact half-step `dt`
coefficient for two accepted evaluations. The native fixture additionally
requires zero covered coarse right faces and exactly two fine groups of 64
faces with duration `0.005`, measure `1/64`, flux `1.2`, and amount `-0.006`
each before checking `q=0.712`. The previous installed native result remains
a red receipt. Source and small header syntax checks do not qualify the fix;
root must rebuild the exact Dim2 SDK and rerun serial/MPI native acceptance.
