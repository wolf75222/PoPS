# Independent reception of the moving output URI/rank correction

Exact corrected production source:
`26e5fb04ebb515e83edcd5a7bfe1e933a96cafcb`, parent `a652f3cf`.
Reviewer GPT-6.1 Sol received it in exclusive checkout
`work/PoPS-sol61-ale-output-rank-reception`, branch
`codex/api040-sol61-ale-output-rank-reception`.

The frozen independent probe commit
`43f0686cc7273c0df62f0ccb619010ea52d939a1` was cherry-picked unchanged as
`f5b87441`. `git diff 43f0686c HEAD --
tests/review/test_sol61_public_ale_independent.py` is empty. The earlier report
remains historical evidence of the two real failures on a652; it is not
rewritten as a positive reception of the old source.

All **39 unchanged independent properties pass in 63.63 s**, including both
previously failing expected refusals. The current exact moving-cartesian-1d
and endpoint-length URIs now require rank one and their explicitly matching
pair, before any endpoint interpretation. The known contract's missing-node,
monotonicity, finiteness and exact volume-difference checks remain active.
An additional public constructor probe confirms an explicit future 2D
coordinate/measure URI pair remains representable. This represents an opaque
future descriptor, not a qualification of a 2D moving provider or its geometry.

The other unchanged properties still receive the actual public
validate/resolve/emission route, scalar/three-component/permuted source
routes, signed relative speeds with a faster moving mesh, four source measure
quadratures, current-provider refusals, detached/archive/writer coordinates,
malformed endpoints and one-ULP measure corruption, and source-byte stability
of the default ledger writer compared with 2a741eff. Twenty-four tiny host
programs execute the emitted physical/relative model, original source CSE,
coordinate law and measure expression using standard math/type scaffolding.
They do not execute native System/Kokkos/MPI or a PDE time advance.

Command, from the exclusive corrected checkout:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I -c \
  'import sys; from pathlib import Path; root=Path.cwd(); sys.path[:0]=[str(root/"python"),str(root)]; import pops,pytest; assert Path(pops.__file__).is_relative_to(root/"python"); raise SystemExit(pytest.main(["-q","--tb=short","tests/review/test_sol61_public_ale_independent.py"]))'
```

No source guard or tolerance was relaxed, no test was removed/xfail/skipped,
and no production edit was made by the reviewer. MAIN/SDK/shared environment
were not changed; no installation, shared JIT, native build, CTest or MPI was
performed. Full public native/restart/rollback and merged MAIN compatibility
remain central reception obligations, exactly as bounded in the historical
independent review.
