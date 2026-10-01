# Independent completed-source bridge and fixture reception

Date: 1 October 2026. Production bridge 18117299ba49d13845948ebccce7100c5429da75, followed by b782cfeb. Fixture 67e14b19150e48e583636813282742871103a271. Review contains only independent tests and this report. No native positive is created.

## Bridge: exact storage prototype and retained tower

181 supplies each level's authenticated storage-owner prototype to the common direct helper. Its retained-byte calculation includes every output level's grown boxes at the selected tuple width, field/layout/Fab metadata, level views and the new prototype-pointer vector. The source tuple and allocation State stay different physical objects. The facade getter is inside the existing fenced/voted closed phase. Direct workspace receives max_bytes minus the maximum rank-local retained reservation; exact integer-word transport remains unchanged. The accepted source is revalidated before publication of the detached output map.

An authentic budget-phase excerpt is compiled with the entire authentic checked Box header under UBSan. Named facade, provider, lane and field-layout substitutes are explicit. Independent inputs use candidate width3, storage width5, selected tuple component1/output width1 and two levels with nonzero ghosts. The expected retained size is computed separately using grown4x5 boxes. Foreign distribution/rank, negative and extreme ghosts, volume overflow and getter failure refuse in the voted phase; a named stale-candidate authority refuses before a new tower object. Getter calls are observed inside the phase. This receives arithmetic and ordering, not the real core's point/lane/epoch checks, distributed collectives, Kokkos storage or true ghost payloads.

The test preserves an exact git-pinned historical181 counter-observation: a max_bytes equal to the retained reservation caused length_error after the tower object had already been constructed. It is not a current-code failure. b782 moves make_shared and prototype-vector reserve behind the complete retained-budget guard. The same actual excerpt now refuses that proposal before construction of any new tower object. Static checks also require output/level-vector reserves and direct output workspace after the guard. Allocator overhead, existing source images, borrowed native providers and process-wide RAM are not newly certified or bounded by this test.

Three SOURCE_ONLY/host tests passed in 2.21 s, including historical/current separation. The earlier independent63f helper reception remains separate: actual full-helper Real32/64, masks/components/ghost zeroing and strict history-contract body. No full native translation unit or MPI/Kokkos execution was run here.

## Fixture: real public authoring, actual outputs and references

The two actual public Case profiles are resolved and emitted from the frozen fixture. Independent assertions compare the unchanged original solve request attrs, input IDs and pre-existing operation prefix against the original physical builder; no interaction is inserted into F. Scalar observes T0/component0, coupled observes T1/component1 of the original T0,T1,z tuple with evolved partition T1,T0. The source remains physically global (block/state_ref/space absent). Q0 is the separate allocation owner; no Q-as-T or dummy physical capture is added. Actual InitialConditionPlan subjects remain canonical and complete. IR19 and both real emitted reduction routes are checked; this is emission admission, not compilation/runtime acceptance.

The declared original equations are H0=T0+T0² for scalar; coupled H0=T0+T0²+0.1T1², H1=T1+T1²+0.2T0T1, and z−0.25T0−0.5T1=0. Original signed/nonsymmetric diffusion, seven Newton controls, FullResidualBasisLU, loads and dt=.01 are reused unchanged. Homogeneous temperatures make spatial flux zero. This fixture cannot qualify nonconstant AMR restriction/flux, a nonlocal term inside Newton or a full M26 PDE.

An independent Fraction contraction oracle receives only explicitly synthetic arrays, midpoint coordinates and finest-cell measures. It separately evaluates W(x,y)=.25+x0*y1−2*y0+.5*x1 over a two-level partition and checks all four raw output contractions against the fixture reference. It distinguishes the transposed kernel, Q substituted for T, wrong T component and inclusion of covered coarse cells. No author accumulation/check_saved result provides these expected contractions. Synthetic tests are SOURCE_ONLY witnesses and never owner-sealed native evidence.

The five runtime phases are initial/accepted/continuous/reloaded/replay. Checkpoint paths include the phase, are hashed immediately, authenticated against the live creator and later checked disjoint from observation files. Raw history slots/sample identities, clocks, auxiliary accepted bytes, geometry masks/native volumes and rank-local state pieces are retained in typed wire and NPZ. Continuous/replay compare exact captured images and authenticated durable payload/lineage. Same compiled component exports CPP and IR, with program hash, sidecar/DSO paths and compilation command retained. Those records alone do not close the cryptographic CPP-to-DSO build graph.

There is no native per-cell I getter. NPZ I_COMPUTED_REFERENCE values are labelled references, and entire_I_per_cell_qualified is false. Only the Program's four scalar reductions are actual I outputs. The scientific check runs on rank zero's metadata; owner state pieces and auxiliary payloads are allgathered, and the collective checkpoint retains per-rank durable diagnostics. Future offline receipt must parse those rank images and actual per-rank JUnit rather than infer independent all-rank diagnostics agreement from the rank-zero check. Private accepted-source leases/stamps are not durable evidence.

Four independent fixture SOURCE_ONLY tests passed in 125.27 s (real resolve/emission plus synthetic Fraction/AST admission). Ruff and diff checks passed. The two marked native cases remain UNRUN. Root must authenticate the new installed SDK, exact source/compiler evidence, complete Serial/MPI JUnit, external owner pins and receipt seals before any native scientific qualification.

## Commands

```sh
rtk proxy env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/review/test_sol61_ir19_bridge_independent.py -p no:cacheprovider --tb=short
rtk proxy env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/review/test_sol61_completed_fixture_independent.py -p no:cacheprovider --tb=short
```
