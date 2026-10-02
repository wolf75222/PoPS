# field-candidate-observation@1

Base e659a29a8292bd7d1e71ceb8231becb9cde7e41d; private implementation after contract review 8743bae. Source/host preparation only. No Native/JIT or full translation-unit compilation was performed here.

## Authority and lifecycle

The private `_enable_field_candidate_observation(1)` capability is off by default. Activation is collective on the authentic package lane before initial bootstrap completes, with idle transaction/macro-step guards and exact version agreement. Invalid Python versions become sentinel version zero and enter the same Native preflight vote, instead of raising on one rank before valid peers enter a collective. The test installs it immediately before the genuine NativeAMRBootstrapConsumer finalizer, without replacing preparation or resolving a Field. An old binary missing the capability fails lookup.

After the genuine Field solve validates, temporary full Field copies and their fence succeed under the existing vote. Capture then deep-copies the successful solver candidate before Ghost. This is precisely the image copied into accepted_potential temporarily and read by the sealed Field storage route (amr_system.cpp:6251); the normal accepted cache is restored afterwards. All diagnostic allocations, mirrors, byte encoding and map insertion are inside one collectively voted capture boundary. A capture failure discards the SolveOutcome and follows the existing whole-owner restoration path.

Only a complete successful Halo preparation swaps the staged diagnostic map into the readable map. A failed new invocation leaves no prior witness masquerading as current evidence. There is no accepted-publication flag: status is producer-completed-consumer-preparation-completed, accepted_publication=false. A later failed outer attempt is not an accepted solution: AcceptedSnapshot::restore explicitly clears both diagnostic maps even for same-time/tick rollback. Clock restoration likewise clears them. The getter refuses a separately authenticated owner stamp, physical point, topology or materialization mismatch; it never compares a consumer clock tick to the owner macro counter. Owner time/macro-step is stamped by cadence completion, while the original consumer LogicalTime remains unchanged. It also refuses an active bootstrap, step, accepted or restart transaction. Current provider/plan/output identities and the complete exact configuration hash are reauthenticated on each read; successful runtime parameter setters/seeding explicitly invalidate both maps. It never calls ensure_engine, materialization or a solve.

Memory retains only the latest invocation per (qualified provider slot, consumer block, consumer level) within one complete hierarchy preparation, replacing the preceding map. No time history or artificial numerical size ceiling is introduced. Memory is O(slots × consumer blocks × levels × full Field storage), with transient capture and getter copies; this optional observation has a real memory cost. Diagnostic storage is excluded from AcceptedSnapshot and all numerical checkpoint members.

## Wire used by the private getter

Each rank-local row has schema pops.amr.field-candidate-observation@1, status and accepted_publication above; provider_slot/provider_identity/plan_identity/output_owner_identity/output_block/output_key/configuration_identity; consumer_block/consumer_level; topology_epoch/materialization_generation and owner_time/owner_macro_step; and exact point fields clock/tick/level/substep/stage/fraction_numerator/fraction_denominator/dt/physical_time/graph_identity/rate_identity/application_identity.

carrier_bytes reuses the existing POPSCAR1 codec as a rank-local archive: one block identity equal to the provider slot, all image levels, authentic patch numbers/distribution ownership, valid and grown boxes, component count and every raw native bit. Dimension, real width, source rank and communicator size are codec authorities. No merged/global image is fabricated. Grown storage capture is not a general Field halo-readiness certificate; OriginalF must use the actual defined numerical supports. An independent reader must authenticate all ranks and geometry before merging replicated rows or evaluating OriginalF.

Persistence writes initial/accepted per-rank .bin plus JSON metadata and actual byte digest before any potential getter. Reload intentionally has no new producer solve witness; the saved accepted witness remains separate from checkpoint-restored cache bit equality. The Field stage cache is not relabelled as endpoint Field. OriginalF guard remains 1e-10 and requires the independent candidate reader adaptation.

## Compatibility and validation

Existing request/component layouts, ABI8 component tables, CP12 and accepted9 numerical contracts are unchanged. This adds exported facade methods and a new diagnostic value type in the existing installed amr_system.hpp; the SDK header signature and native DSO must be rebuilt/authenticated. Old binaries cannot provide this optional capability. No new standalone header was added.

Source/host command:

    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider -o pythonpath=python tests/review/test_sol61_field_candidate_observation_source.py tests/review/test_sol61_initial_field_ghost_native_preparation.py tests/review/test_sol61_accepted_initial_field_point.py tests/python/unit/amr/test_accepted_halo_preparation.py -q --tb=short

Final coherent cohort: 44 PASS, 20.09s, zero skips. One public Native fixture node collects successfully (not executed). The tests compile the unchanged getter body against explicit Source-only owner stand-ins, admitting a distinct consumer clock tick and refusing stale owner/configuration/provider/transaction authority. Actual POPSCAR1 encoding/decoding runs in dimensions 1/2/3 with arbitrary slots, components, grown shapes, exact -0/subnormal bits and malformed shape/owner/truncation refusal. Binding-vote and rollback/default guards are checked against Source. These checks do not prove full runtime translation-unit compilation, Native solver, MPI votes, Field freshness or device execution.

Packaging manifest: PASS, 172 api / 7 abi / 19 sdk-root / 178 sdk-support / 8 test-only. Root owns full-TU compilation, immutable SDK rebuilding, actual bind/run and independent scientific reception. The SDK13 failed receipt remains failed.
