# Independent Source review: warm start versus Halo endpoint

Inspected clean Galileo checkout `PoPS-sol61-accepted-field-cache`, HEAD e659a29a8292bd7d1e71ceb8231becb9cde7e41d. No production edits or Native execution.

The public header describes restored field potentials as an all-provider warm-start image (`include/pops/runtime/amr_system.hpp:1005`). The getter calls materialize_field and gathers accepted_potential (`src/runtime/amr/amr_system.cpp:17085–17100`); it is not a promise to solve at the current Q endpoint.

The Halo dependency path performs the real named Field solve at the authenticated candidate point, records accepted_halo_producer_point, copies the solved candidate into temporarily exposed Field/provider storage, and consumes DiscardCandidate (`9950–10038`). Whole-Halo restoration restores accepted Q, Field warm-start images, provider groups and publication registries, and clears the temporary producer witness (`14169–14191`). Restoration runs on success as well as failure (`14241–14247`). The actual boundary preparation follows Field preparation and producer preflight before its copy/fence (`14210–14218`).

Therefore a FE stage-t_n warm-start image after endpoint Halo is consistent with this implementation. An endpoint OriginalF oracle over that getter tests a different image and cannot establish failure of the actual Field consumed by Ghost. Publishing endpoint Field into the accepted cache would change the deliberately temporary semantics and requires an explicit separate contract.

The existing raw Ghost expression gives evidence of consumed boundary-strip values at its point; it does not export every active Field cell needed for the full composite OriginalF residual. A complete residual proof needs a generic, qualified-slot/level/exact-point read-only image of the solved candidate, captured after successful solve/copy/fence and before Ghost consumption, without a getter solve or accepted publication. That observation extension is proposed only; this review does not implement or qualify it. No Native/MPI/scientific pass is inferred from these Source paths.
