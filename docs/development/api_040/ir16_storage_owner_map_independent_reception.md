# IR16 storage-only block routing: independent source reception

Candidate `a02e151d4f6c393d1193f442c1a9eb173a825fd8`, parent
`378f29084c915ae316308481949ea174858197b2`, 1 October 2026.
The production diff is nine lines in `serialization._block_indices`: authenticate
original global-history issuances, then append previously absent exact owners
in sorted ring-name order. Existing physical and readonly dt-bound indices keep
their positions. No SSA State node, physical capture or commit is created.

The independent public witness is the previously frozen `_storage_only(5)`
Case/Program. Its fourth block has an issued five-component TimeState and real
numerics/initial condition. The scalar FieldProblem observation keeps
`block/state_ref/space=None`; its source tuple and solve remain unchanged. A
three-slot ring is declared with that exact owner. Running the authentic parent
index body on this same Program omits the owner; candidate appends index 3.
The candidate's genuine AMR shape emitter then emits the originally issued
IR16 descriptor. Nothing is inferred or minted from checkpoint JSON.

Source probes exercise raw/frozen/to_graph routes, exact indices, original
registration and point/clock, no physical promotion, repeated ring names,
mutated registration owner, boolean width, foreign clock, removed qualification,
removed store, cloned owner and removed TimeState scope. Every mutation must
refuse before returning a route. Source-only Case/validate/resolve and actual CPP
emission run for Uniform and AMR. The four routes are thermal, matter, spectator,
storage-only. Only the three pre-existing physical sources are committed to make
the Program executable; the storage-only TimeState has no State.n read or commit.
The physical owner width five is distinct from the scalar ring width one.

Four fresh legacy profiles (linear/nonlinear, StagePoint/TimePoint) are compared
in new interpreters against parent source extracted from local git. Both sides
execute exactly the same helper file and callsite. Comparison retains entire
IR with actual provenance, IR hash, entire CPP, CPP hash and three Module hashes;
there is no normalization, removed field, static checkout-path golden, fetch,
installation or compiler invocation. This explicit review test needs the local
historical object; it is not a shallow-CI unit-test requirement.

Command:

```
rtk proxy env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 \
 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest \
 tests/review/test_sol61_storage_owner_map_independent.py \
 -q -p no:cacheprovider --tb=short
```

**Final: 17 SOURCE_ONLY PASS in 99.36 s; Ruff check/format and diff-check pass.**
One Python 3.14 tar-extraction deprecation warning concerns the explicit local
git source archive; no test failure or Native execution is hidden by it.

This receives Python authoring/resolution/emission and route authority only.
It does not receive compile/bind, Kokkos execution, native checkpoint capture,
AMR initial regrid or MPI. In particular ROOT's real post-metadata bind refusal
at the old scalar-output capability is a separate contract gap being ported as
capability @2. No runtime outcome from that red campaign is turned positive by
these source checks. No production file, header, installed environment or ROOT
artifact was modified by this review.
