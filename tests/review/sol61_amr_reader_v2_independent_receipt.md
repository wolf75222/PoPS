# Independent bounded review of AMR scientific reader @2

Reader reviewed: `bcf304c`, private worktree `PoPS-sol61-amr-reader-v2-review`. Status:
**SOURCE_ONLY, no correctness blocker identified**. This is a review of saved-image verification,
not native checkpoint/restart acceptance. No MAIN, Native worktree or environment was modified.

The independent reader's POPSCAR1 little-endian words match the native codec: dimension/scalar
width/rank/shard/levels, opaque block order, ordered block-level-global-patch records, signed owner
and valid/grown coordinates, full component-major Real bits. Its ExactContractBuilder reproduction
matches actual C++ framing: tag plus big-endian uint64 length; signed integer tag and native width;
float/double tags plus unchanged IEEE bits. Valid/grown axes and x-fastest storage order match
`append_rank_local_carrier_rows`. Local indices follow ascending global patches filtered by current
rank, including the full index sequence on every replicated rank.

The scientific receiving path binds the complete state inventory to checkpoint patch boxes,
owner maps, block component declarations and valid scientific arrays. It reconstructs expected
full-storage hashes for every rank. The full reader additionally checks registry files against
externally pinned file hashes, compares accepted/reloaded and continuous/replay registries, and
retains the historical @1/CP11 reader separately. Standalone `decode` is deliberately not full
scientific reception: source completeness/expected geometry are closed by `receive_carriers` plus
the surrounding topology/envelope checks.

## Additional independent oracle

The C++ host emits POPSCAR1 using the actual codec and hashes using the actual
ExactContractBuilder and SHA256 implementation. Python compares those independently emitted
hashes, rather than constructing both reference and observation with its own hash implementation.
Two opaque blocks, one/two components, two global patches, both source ranks and both partitioned
and replicated ownership pass. Ghost signed zero and a quiet-NaN payload remain exact bits.

Inner archive changes (even if an inner file seal is recomputed) are rejected against the immutable
independently emitted registry: ghost/valid bits, owner, grown bounds with unchanged extent,
non-uint8 archive type, truncation, illegal binary32 high bits and rank routing. Rank-one row
omission is rejected in both distribution modes. These are explicitly **synthetic wire-frontier
oracles**, never ROOT scientific/native acceptance evidence or externally approved campaign seals.

Command actually run:

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest -q --noconftest -p no:cacheprovider tests/review/test_sol61_evolved_stage_amr_saved_reception_v2.py tests/review/test_sol61_carrier_reader_cross_independent.py
```

The cross oracle compiles only its small STD host program with C++20 `-Wall -Wextra -Werror`;
no PoPS import, native DSO, Kokkos, JIT or runtime TU is used. The test asserts PoPS was not imported.

Principles 1.1/1.3: recorded physical state and native storage are authorities; opaque-name and
component variations exercise the shared wire mechanism. Principle 1.6: actual compiled host
codec/hash proof is explicitly separated from real ROOT execution. Principle 1.7: no native
performance conclusion is claimed. Principle 1.8: no production abstraction or reader relaxation
was introduced. Actual ROOT reception still requires real saved states and external owner seals.
