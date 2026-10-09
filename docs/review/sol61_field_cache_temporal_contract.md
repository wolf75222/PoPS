# Field cache versus accepted Ghost point authority

Source review: e659a29a8292bd7d1e71ceb8231becb9cde7e41d. No production change or new Native qualification.

## Confirmed contract

`restore_field_potentials` documents the complete all-provider warm-start image (include/pops/runtime/amr_system.hpp:1010). `field_potential_level_global` materializes storage and reads accepted_potential, without a new solve (src/runtime/amr/amr_system.cpp:17057). The different `named_field_values` API explicitly solves at pops.amr.direct-field-read (17119); its current-value documentation does not establish endpoint authority for the cache getter.

The last FE stage solve at t_n and temporary accepted Ghost solve at t_n+dt are distinct. prepare_accepted_halo_field_dependencies uses genuine candidate images for Ghost, while prepare_accepted_halo_candidates restores accepted Field/provider images and provenance on success as on failure (9949,14169). A stage cache near 2 and endpoint Ghost based on 2.03125 are compatible. The endpoint OriginalF assertion on the stage cache is misplaced. Publishing endpoint caches or changing FE would introduce a new obligation without an existing contract.

## Observation and minimal extension

A PopsQualifiedConstFieldV1 Ghost dependency is not a complete hierarchy image. The physical path packs region.numPts cells through pack_dependency_region (include/pops/mesh/boundary/prepared_boundary_component.hpp:777,1359). A test-owned wrapper around the real Ghost table can attest exact consumed boundary strip, qualified identity, geometry and LogicalTime; it cannot reconstruct volumetric OriginalF with interior stencils.

The smallest complete proof needs a versioned, non-mutating producer witness capturing the successful real candidate Field image, owned levels/boxes/components, qualified provider/output identity and exact BoundaryEvaluationPoint before Ghost consumption. Keep last-stage cache images separately and preserve all physical/cache rollback. Diagnostic allocation must converge under existing votes before publication; MPI evidence authenticates rank/ownership. No getter refresh, additional solve or endpoint cache publication is needed.

This witness is proposed, not implemented. The immutable SDK13 failure remains failed; genuine Ghost captures precede getter observation.
