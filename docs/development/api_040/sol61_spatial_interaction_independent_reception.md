# Independent reception of the spatial interaction port

This receipt concerns the finite midpoint direct interaction port, not the complete M26 aggregation–diffusion equation. No native library, JIT, MPI process, GPU, or scientific archive was executed or certified by this reviewer. The parent agent owns those receptions and external scientific seals.

The review worktree is `/Users/romaindespoulain/dev/tmp/pops-sol61-spatial-nonlocal-review`. The initial exact candidate was `dd46ea495f01f0e8e59e0caa3f9b8f697483f9b2`, including its earlier global-history storage dependency from base `bf1b2cfb34bf047f2c6b5afb599dd2187235a58d`. Only the two independent test files and this report are changed in the review commit.

## Counterexamples and correction reception

* On dd46, a genuine builder-issued `State.n` record replaced with `point=u.next.point` was admitted by the public interaction authoring, contract and emitter. It emitted `ctx.set_stage_time(1,1)` while its native read still addressed `ctx.state`. Both issued and accepted scopes admitted it. Three malformed kernel trees also serialized as IR17 even though the emitter rejected them.
* Correction `61c78b9b430f3101c49f1a9ec306e2936edf78fc` refused those five injections. The same mutation hidden by `p.value("candidate", 2*source, at=u.next.point)` still passed; mutating a canonical leaf after issuing the candidate also passed.
* Correction `6a159b34ee72b9a8ebc29c95e79c32175aff9999` closed that input closure. A public `p.branch(p.requested_dt()>0, true_fn, false_fn)` then exposed the remaining region closure: the n-read existed in `true_block/false_block`, while branch inputs held only the condition. The original and canonical leaf mutations were admitted.
* Candidate `3d0048e96e372fcf244980216f987687b5bf1928` independently authenticated raw replica masks before inclusion. Its point region hole remained. The additional serialized-unit injections admitted a boolean exponent, a zero exponent, an unreduced `2/4` exponent, and `coordinate_units=(None,None)` despite the typed measure authoring refusing unknown entries.

The tests preserve genuine calculated next-point candidates and branches as positives; they do not reject actual arithmetic progression. The dimensioned positive uses two different input dimensions, a length-inverse kernel, two length coordinate measures, and component order `(1,0)`, checking the exact output dimensions independently.

Final correction `13a3cabe5c1d6830e3d6a9fe3357c945d243c0eb` closes all six final injections. The independent suite receives **24 PASS**, including genuine calculated/branched candidates at next, exact dimensioned component permutation, and both native-Real-width actual-body host compilations. The combined author and independent selection receives **71 PASS in 25.30 seconds**. Ruff and diff checks pass. No production file is changed by this review.

Raw replica active/coverage/kappa values are now compared independently before source inclusion, including excluded cells with zero effective measure. The final transport selects a uint32/uint64 word from the actual Real width; 16-bit limbs preserve sign-zero and finite words without a floating SUM normalization. The actual MPI divergence/empty-rank and transport runtime still require native reception.

## Independent actual-body host check

`test_sol61_spatial_interaction_reception.py` appends the complete current `spatial_direct_interaction.hpp` body unchanged (only preprocessor includes/pragmas are omitted) and the real `FiniteCompensatedSum` definition to `sol61_spatial_interaction_host.cpp`. `/usr/bin/clang++ -std=c++20 -O0 -DPOPS_REAL_TYPE=double` and a separate `-DPOPS_REAL_TYPE=float` build compile and execute that translation unit. The output width is asserted as 64/32 bits respectively, so a duplicate double build cannot masquerade as float coverage.

The named substitutions are Kokkos View/mirror/deep_copy, cell reductions, MultiFab/Fab/view/Box/Geometry/distribution and a single-rank execution lane/collective. They deliberately provide no claim about actual Kokkos allocation, device launches, MPI agreement, facade ownership, callbacks, or native runtime transactions. The checked algorithm is the actual consumer body, not a second implementation of the interaction.

Twenty-seven checks per Real width cover dimensions 1/2/3, width-three permutation `(2,0,1)`, a signed nonsymmetric coordinate kernel, cell volume and fractional kappa, covered/inactive NaN exclusion, replicated-mask transport on a single lane, exact IEEE minus-zero extraction/transport, local nonfinite source/kernel/mask refusals, duplicate/out-of-range components, workspace and checked arithmetic overflow. A two-level partial tower has a covered coarse NaN and fractional fine kappa; its surviving coarse target is exactly `2.75`, while its covered target is zero. Target results are not multiplied by the target measure.

The two-dimensional manual values use dyadic coordinates and constants. At the first active target, output components are exactly `(0.375,3,-4.6875)`; at the second they are `(1.5,1.875,-1.875)`. These closed arithmetic checks do not import an author numerical oracle.

## Legacy and commands

A fresh Git archive of bf1b and the candidate were passed to the same source emitter callsite. Four public legacy witnesses (mixed linear, its component permutation, implicit stage, nonlinear-map implicit stage) retained all eight exact IR/C++ hashes. The final local JSON is `outputs/nonlocal-review/fresh-legacy-parity.json`, SHA256 `86debf870bd46d178e3d96276d62ba8a0ab99a061f251671cb369a1639a8e088`; source parity is distinct from a built native artifact.

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q -o pythonpath=python tests/review/test_sol61_spatial_interaction_reception.py --tb=short
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q -o pythonpath=python tests/review/test_sol61_spatial_interaction.py tests/review/test_sol61_spatial_interaction_reception.py --tb=short
rtk proxy ruff check tests/review/test_sol61_spatial_interaction_reception.py
rtk git diff --check
```

## Open native obligations

The author suite was replayed from the frozen final candidate, including its actual-header host checks (61 assertions per Real width) and genuine Uniform/AMR emitted C++ syntax. The actual AMR engine host test from the author was replayed separately; it exercises coarse candidate commit before fine and keeps accepted carriers distinct, but it does not execute the actual AmrSystem facade/DSO. Native reception must check that routing, history retained points and replay/regrid, source and target ownership, replicas and empty ranks, divergent raw masks and source bits, rank-local budget/allocation/nonfinite failures, and refusal before physical publication. The detached scratch node preparation was reviewed as allocating before consensus and transferring/replacing after the vote without allocation or throwing move assignment; actual rollback still needs native reception.

Multi-level issued AMR candidates remain refused because no simultaneous composite candidate barrier exists. This is an unresolved realization seam, not a general restriction on physical kernels. Cartesian midpoint geometry is the current provider; moving geometry is not silently substituted. Scalar W applies separately to arbitrary selected ordered source components; a matrix-valued cross-component interaction is not supplied here. No symmetry, positivity, SPD closure, periodic-image convention, diagonal exclusion or model-size cap is inferred.
