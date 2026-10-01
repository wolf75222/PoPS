# Nonconstant AMR reception prepared, Source only

Base: `05159edd`. No native compilation, JIT, installed package mutation or
scientific native execution is performed by this preparation.

The unchanged public case is cells8, width2, two ratio-two AMR levels, periodic
full-y strips. Its nonlinear conserved quantities are
`Q0=a+a²+0.1b²`, `Q1=b+b²+0.2ab`; its constraint is
`z=0.25a+0.5b`. The constitutive matrix remains
`[[0.012,0.002],[-0.001,0.014]]`. OriginalEvolution, seven Newton controls,
FD step `1e-6`, acceptance `3e-8`, timestep `0.01`, threshold `1.03`,
TagBuffer0, all scientific assertions and the public builder are preserved.

The historical spatial reader/profile @1 remains unchanged. New
`tests/review/sol61_evolved_stage_amr_spatial_reception_v2.py` qualifies only
`nonconstant-original-composite-Q-tag-selection-periodic-strip@2` with separate
owner pins/ROOT approval @2. It uses current homogeneous reader @3 strictly for
wire/identity contracts: ROOT-attested NativeABI6, CP12/POPSCAR1, accepted8,
TagSelection1, Program IR controls/provenance, current history and temporal
cursors, Run CBOR with raw digest bytes. Its scientific computation remains the
independent nonconstant leaf-flux oracle. No homogeneous archive qualifies this
profile.

Capture saves unchanged native all-rank carrier manifests at initial, accepted,
continuous, reloaded and replay, with a sealed JSON registry. Full valid patch
geometry is derived from the actual state manifests, with indexed completeness
and cross-block/replica consistency. `native_patch_boxes` retains the public
native refinement-only boxes; `carrier_patch_boxes` closes the base level too.
The reader independently reconstructs complete geometry from CP12 and compares
both forms. It never appends an invented coarse box. Raw observations and the
registry are saved before scientific checks.

The reader checks nonlinear restriction `average(H(fine Psi))`, not
`H(average(Psi))`. After ordinary science passes, it measures signed component
Jensen gaps on authentic accepted/continuous snapshots and injects the latter
countermodel offline. Reception requires its refusal at the existing covered-Q
guard `3e-8`; nonzero gap `>1e-10` alone is insufficient. Q1's gap can be negative
or cancel, unlike Q0's positive variance combination. Source tests cover this
distinction, signed/transpose/cross-flux mutants and the unchanged guarded
countermodel. Quadratic coarse-fine ghost interpolation concerns point values
at offsets ±1/4; it does not redefine conserved Q cell averages.

Source/offline tests read retained genuine homogeneous ABI6 archives only to
exercise protocol/geometry attacks and verify refusal as nonconstant. They do
not provide nonconstant Native evidence. Existing historical math tests remain
included. Reproduce from this checkout:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
  /Users/romaindespoulain/miniforge3/envs/pops/bin/python -m pytest --noconftest \
  tests/review/test_sol61_evolved_stage_amr_spatial_reception_v2.py \
  tests/review/test_sol61_evolved_stage_amr_spatial_reception.py -q --tb=short
```

ROOT owns installed fixture execution, all-rank JUnit, file inventories,
native/SDK/package/ABI receipts, owner pins and approval seals. The command
`python tests/review/sol61_evolved_stage_amr_spatial_reception_v2.py contract`
prints the required pin contract. Future native failures must retain raw images,
CPs and carrier registry before diagnosis; no equation, FD step, tolerance,
threshold or scientific guard is adjusted by this preparation. Arbitrary 2D,
3D, GPU, variable constitutive matrices and native nonconstant reception remain
unqualified.
