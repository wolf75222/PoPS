# Hooke independent final BVH review

Actual reviewer GPT-6.1 Sol, independent of codec/BVH author Banach. Frozen baseline `1b9ca827c76cabcf0eec83963376c809a68eeca6`; tests/docs-only review fork. Earlier reviews `30bdfc89` and `4562ee3c` preserved.

Decision: no source correctness blocker found before native compilation. The earlier unconditionally quadratic aligned-grid overlap concern is addressed by the common balanced BVH. Native mandatory acceptance remains open, including exact full-carrier restart hashes and MPI/device failure rollback; no native DSO was rebuilt or loaded for this review.

The exact closed-box intersection remains authoritative: a leaf can reject a source box only when one dimension is disjoint, and every unordered pair has one query with a larger leaf index. Internal nodes conservatively enclose all descendants. Median halves bound recursion by O(log P); recursive node writes retain indices rather than invalidated vector references. std::midpoint avoids signed endpoint addition; uint64 subtraction gives the exact nonnegative span for ordered int64 endpoints, including ranges crossing zero. Cached block-zero per-level ranges preserve exact patch counts, owners and geometries. Scalar component checks now scan rows once and their block indexing is safe after decoder validation. All three exported methods, explicit native instantiations and pybind bindings remain present. Source/target full payload copying and valid-bit oracle are unchanged by this performance patch.

Host author oracles actually executed with C++20, -O0 -Wall -Wextra -Werror: Dim1/2/3 spatial oracle, then original codec oracle, both passed. Independent review adds 750 deterministic geometries with variable box side lengths, compared with an independent exact all-pairs closed-box oracle; passed. No heavy production translation unit, JIT or environment setup.

Author performance evidence was read, not rerun: same byte-identical computation, 65,536 patches/block, old/new median host milliseconds 1575/9.6 (Dim1), 2048/23.9 (Dim2), 3117/36.3 (Dim3). Index overhead and host binary __text delta are disclosed. Those figures do not establish production performance, MPI/GPU cost or a universal complexity bound. Irregular BVH geometry can still cause many candidate visits; no patch limit or relaxed scientific oracle was added.

| Principle | Final bounded decision |
|---|---|
| 1.3–1.5 | Common closed-box geometry and byte-persistence effect; no model dispatch |
| 1.6 | Real independent host oracle passes; native restart acceptance open |
| 1.7 | Previous aligned-grid P2 addressed with measured host tradeoff; pathological geometry caveat retained |
| 1.8 | Reused one codec and common index; no per-model expansion |

Reproduction:

```sh
/usr/bin/clang++ -std=c++20 -O0 -Wall -Wextra -Werror -I include tests/review/hooke_state_carriers_bvh_host.cpp -o /tmp/pops-hooke-bvh-independent
/tmp/pops-hooke-bvh-independent
/usr/bin/clang++ -std=c++20 -O0 -Wall -Wextra -Werror -I include tests/review/sol61_state_carriers_spatial_host.cpp -o /tmp/pops-hooke-bvh-author
/tmp/pops-hooke-bvh-author
/usr/bin/clang++ -std=c++20 -O0 -Wall -Wextra -Werror -I include tests/review/sol61_state_carriers_codec_host.cpp -o /tmp/pops-hooke-codec-final
/tmp/pops-hooke-codec-final
```

Independent stdout: HOST_ONLY BVH: 750 variable-sized closed-box geometries agree with independent all-pairs.

Author spatial stdout: CSV header only (oracles pass; benchmarks disabled without OLD_HEADER). Codec stdout: SOURCE_ONLY codec: ghost bits, signed zero/NaN, N components, shards and refusal cases passed.

Inspected SHA256:

```text
ab8067d5ca6d813b462d4c831e53bfde752488d3b87f9cab6678f7aab26209dc  include/pops/runtime/checkpoint/state_carriers.hpp
cb01d980db60d5287efda954db2f6c3313578b495228629978b78c8f9a19a813  src/runtime/amr/amr_system.cpp
67d3489916ac6a1e4da5b0dc819e827577a6a8128fc33e971559469ceb5cdea3  python/bindings/core/init/init_amr.cpp
78b37ea9de0183b429f3fab2c3ab88a0b9f7ba978dff67e7d455de51bb97c45e  tests/review/sol61_state_carriers_spatial_host.cpp
f1eb016a5616793fbba991ff03636cd79c7f840061ad8303ba924f086afbd453  tests/review/sol61_state_carriers_spatial_performance.md
```
