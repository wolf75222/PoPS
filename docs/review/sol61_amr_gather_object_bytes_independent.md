# Independent Source review of AMR gather object-byte fixtures

Reviewed author 97d37e85b23a92b4ec6fa90bec99173c1ab272da and followup aa212e9b31057d7f9e2a94cfbf002021e3316af0. Private review copied only these tests/docs patches on the integrated Source baseline; author, ROOT Native, ENV and ROMEO were not modified. No Native build/import/run was performed. Verdict: ready for independently authenticated engineering Native execution, not Native received or scientifically qualified.

The first revision checked only levels/ranks before iterating the archive block list. A whole omitted block could therefore evade that getter comparison despite the authenticated three-model partition. The followup closes this test coverage gap on both initial and written archives: exact dimension2, Real64, global shard−1, two levels, actual world size and ordered Q0/Q1/forcing partition are mandatory. Getter NPY dtype and shape are explicit before object-byte equality. This was a fixture gap, not an observed Native omission.

Actual AmrSystem binding signatures for coarse_local_boxes, set_block_level_state and block_level_state_global were checked in init_amr.cpp. Public validate/resolve uses the original evolved AMR model and three real PatchLayout choices. Compile uses C25 require, and the existing retained-provenance helper authenticates all three Model sources, exact compiled partition and retained Program CPP/IR before bind. Writes use the existing engineering setter on actual owned valid Fabs; no physical evolution or interpreter-cell backend is substituted.

Coverage is precise: all three policies require partial fine coverage on each actual block and getter equality to native full-carrier valid bits. Uncovered fine cells must be positive zero; actual covered negative zero must remain distinguishable bitwise. Replicated coarse requires owner−1. Distributed coarse requires nonnegative owners. MPI2 partitioned requires both coarse owners nonempty; MPI2 empty-owner requires one empty coarse owner and one nonempty. Serial cannot qualify those two-rank conditions, and no assertion proves an entirely empty engine/fine rank. Full-grown archives are retained, but the getter oracle compares valid cells only: no Ghost formula/readiness, evolution, rollback or restart qualification is claimed. Clock/macro/temporal relations must remain unchanged across the engineering writes.

Independent synthetic wire tests use a separate struct encoder and unequal x/y valid offsets, two components, finite dyadic values and negative zero. They reject forged owner/shard, truncated/trailing archives and invalid dtype/rank; they distinguish grown-only mutations from unchanged valid data without inventing a Ghost claim. These synthetic archives are Source probes, never Native evidence.

Validation: seven author Source tests passed in48.35s, including three genuine public resolutions. Eight independent codec/bit probes passed in0.11s. After the followup, ten affected author pure tests plus eight independent tests passed in1.07s; the unchanged three public resolutions were not repeated. Three future Native nodes collected in1.26s. No skips; three deliberate deselections in the final targeted replay are the previously received resolutions.

Pinned external JUnit:

- `/tmp/sol61-amr-gather-independent-source.xml`: 3b0e01e7702f926e5373de70b2d0e2444e7150fe2ae608b8e48ed125c066a27a
- `/tmp/sol61-amr-gather-independent-adverses.xml`: fa5e61c794c9dd2e294004d1974b4c7fac6a46af8b799d5f15b8a4c7deade2f0
- `/tmp/sol61-amr-gather-independent-final.xml`: 39bdcd3ee4214efac9e646f91b197c2ba204c0907dd54f901ab4a298a2941ec9

ROOT still must receive genuine Serial/MPI2 launcher/XML/installed package/DSO/SDK identities and every actual carrier/NPY/provenance file. No external seal or automatic approval is emitted by this review.
