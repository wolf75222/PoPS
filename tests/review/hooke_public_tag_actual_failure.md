# Independent public tag Native failure review

Hooke GPT-6.1 Sol. Parent8ba260e3; docs only, no PoPS/native/JIT/build/environment/Main changes. Actual closed archive `/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdkbb416-public-tag-serial-dim2`.

Principle→decision: preserve authentic failures and strict full-grown constant1 guard. Actual result failed rc1,8tests/8failures/0errors/0skips. Raw XML8testcases and log captured; result log/identity SHA pins independently rehashed. All failures occur at test_public_tag_selection_native.py67, values[0] == ones full grown carrier, after successful earlier coarse physical tag, fine coverage, masks, nlevels, acceptedstep, unchanged boxes, carrier decode and finite value checks. Thus proposed base-row/fine-only oracle defect is falsified for this actual test batch: receive() already ignores nonfine rows and works with actual fine-only runtime.patch_boxes.

Actual first linear Buffer0 shape8x8 failed vector is100grown cells (10x10): outer ghosts0 and interior valid1 are visible in log, versus strict desired1 everywhere. The log formatting also prints misleading100/100 mismatch count despite visible interior1; do not rely on that count to derive validity geometry. No raw carrier image was saved by the fixture before failing; no offline fullbit reclassification is claimed. Each remaining combination (linear/injection,Buffer0/1,8x8/8x12) fails at the same guard.

Source ordering/capture: src/runtime/amr/amr_system.cpp11073–11104 checkpoint_state_carriers fences and copies each actual Fab host scalar value unchanged with bit_cast. include/pops/mesh/storage/field_view.hpp27 is component*component_stride plus celloffset; tests/review/sol61_amr_full_carrier_offline.py108 uses same component-major grown layout, so reshape(2,-1) in test is correct. Capture performs no field refresh intentionally. Changing reader/capture to fill ghosts would manufacture evidence and conceal the actual accepted carrier contents.

Source causal seams to examine: include/pops/mesh/storage/fab.hpp58–69 allocates grown storage and zero-initializes it; src/runtime/amr/amr_system.cpp13731–13732 stages publication from accepted carrier then copies active valid cells; include/pops/runtime/amr/prepared_multiblock_hierarchy.hpp787/1480 publishes via full storage deep_copy. These paths can preserve or publish stale ghost bytes; this review does not yet establish which initial/prepared Program ghost-fill consumer was omitted for the stationary zero-flux expression, or whether the stale bytes originate initial bind or candidate publication. Exact observed defect is accepted fullgrown constant preservation, not tag selection. Production owner must inspect actual phase/ghost authority before fixing; no speculative production edit here.

No guard, threshold.7,shape, Buffer0/1, transfer policy, stationary equation or grown constant1 assertion was changed. Useful next capture-only instrumentation: save checkpoint_state_carriers raw bytes immediately after bind and after accepted run (outside active transaction), plus exact geometry/phase to distinguish initial missing fill from publication overwrite; keep failing assertion strict. This would add observation only, not repair it. Current archived failure retained.

Actual log SHA256 e579e9dd94f408bcfeb6fab2914bb9f2e979013885c90c75f2b9b8ce255c0596; identity SHA256 3e448c93e31dcbad8b6f4b2558f23f7d13205d8669dbe9ab0818cc665209280c. Source and tests above read in private detached parent; source-file hashing listed below.

tests/python/integration/amr/test_public_tag_selection_native.py SHA256 042bbb7502d3805fa7b64009ae7072e2a948aec52b0057b613bb47db82d2ce2b
tests/review/sol61_tag_selection_oracle.py SHA256 fe2ebaa22cfad42efe45e321f75417449ff085e3d66a12c468350de28ec5b68f
tests/review/sol61_amr_full_carrier_offline.py SHA256 c5148a74fb067087f4c157c38d5a763e85169089c988b8a302c1d93720ad5e9d
include/pops/mesh/storage/field_view.hpp SHA256 42b34c920f03bddb39bff85d1d16c2098ab4647d52b158101e3e60237f564b36
include/pops/mesh/storage/fab.hpp SHA256 7ed3555c927cfe336eafa841b268770d57a30650b2b454abf85bb087e5bb26ac
include/pops/runtime/amr/prepared_multiblock_hierarchy.hpp SHA256 caa23b2c5f22ee5a555d855e798e47f7322376dfd7ba25530271069a30a60c57
src/runtime/amr/amr_system.cpp SHA256 d372fa562671830ede3f6ac3873022c67a4728055116738c00bda8103592caa6