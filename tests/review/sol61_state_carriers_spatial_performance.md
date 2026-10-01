# POPSCAR1 source spatial-validation performance slice

Status: **SOURCE_ONLY**. No native runtime translation unit, binding, JIT or installed DSO was
built. Author worktree: `pops-api040-full-carrier-fix-20261001`; baseline `86a71b6`.

The mandatory source-patch overlap check now uses a balanced STD BVH over closed int64 boxes.
`std::midpoint` avoids signed coordinate overflow; unsigned subtraction compares bounding spans.
Every leaf candidate still undergoes the exact closed-bound test. Cached block-zero level ranges
replace repeated archive scans and per-patch binary searches for all subsequent blocks. There is
no empirical rank/patch limit, coordinate narrowing or change to POPSCAR1 bytes.

The host oracle covers Dim 1/2/3, disjoint regular geometry, overlaps and touching closed-cell
bounds, both signed int64 extremes and an index spanning them, multiple source levels, owner
mismatch, plus 100 deterministic geometries per dimension compared against exact all-pairs.
A complexity counter protects regular geometry against returning to quadratic candidate growth.
The earlier codec oracle continues to cover ghosts, signed zero/NaN payloads, shards and failures.

## Measured comparison

Apple clang 21.0.0, C++20 `-O3 -Wall -Wextra -Werror`, arm64 macOS, one host thread, three runs.
Each old/new pair uses byte-identical input: two blocks, one level, two components, ghost depth
one, one valid cell per patch on a regular grid with cell spacing two. Old code is the exact
baseline validator with one added pair counter. Fixture creation, encode and decode occur outside
the timed validation phase. Times are medians in milliseconds.

| Dim | P per block | Old ms | BVH ms | Old pairs | BVH node tests | Extra BVH storage bytes |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 5.232 | 0.504 | 8386560 | 96256 | 425936 |
| 2 | 4096 | 9.054 | 1.189 | 8386560 | 96256 | 556992 |
| 3 | 4096 | 9.953 | 1.381 | 8386560 | 96256 | 688048 |
| 1 | 16384 | 96.589 | 2.095 | 134209536 | 450560 | 1703888 |
| 2 | 16384 | 130.978 | 5.166 | 134209536 | 450560 | 2228160 |
| 3 | 16384 | 156.823 | 7.264 | 134209536 | 615362 | 2752432 |
| 1 | 65536 | 1575.110 | 9.600 | 2147450880 | 2064384 | 6815696 |
| 2 | 65536 | 2048.490 | 23.882 | 2147450880 | 2064384 | 8912832 |
| 3 | 65536 | 3116.850 | 36.312 | 2147450880 | 2809724 | 11009968 |

No intersecting leaf pairs occur in these disjoint grids. Index storage counts the actual vector
capacities for nodes and ordering, excluding payloads and the small source-level range cache.
Combined benchmark process peak RSS was 629637120–631324672 bytes; it includes both decoded
archives, encoded bytes and all oracle/benchmark work, so it is not attributed to either validator.
For a matched Dim2 decode+validate host probe, old/new executable files were 58136/58488 bytes;
`__text` was 8360/13492 bytes. These are host probes, not production binary-size measurements.
A BVH can still visit many candidates for irregular enclosing boxes; these measurements establish
normal aligned-grid growth, not a worst-case geometric complexity guarantee or native MPI/GPU cost.

## Reproduction and frozen evidence

```sh
rtk proxy python3 tests/review/sol61_state_carriers_spatial_benchmark.py --directory /tmp/pops-state-carriers-bvh-evidence --runs 3
rtk proxy c++ -std=c++20 -Wall -Wextra -Werror -Iinclude tests/review/sol61_state_carriers_codec_host.cpp -o /tmp/pops-state-carriers-codec-host
rtk proxy /tmp/pops-state-carriers-codec-host
```

Saved report, CSVs, resource receipts, compiler commands, source SHA256s and matched size probes:
`/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/state-carriers-bvh-host-20261001/`.
Production acceptance still requires independent review, native compilation and the complete
public checkpoint/restart/replay cycle with strict carrier equality.
