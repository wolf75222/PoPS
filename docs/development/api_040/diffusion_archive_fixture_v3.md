# Retained diffusion checkpoints and compiled Program IR

The actual SDK2e4 frozen-D@2 and candidate-D@1 campaigns pass their native
in-process saved-state/restart/replay comparisons in Serial and on each of two
MPI ranks. Independent archive inspection exposed a separate fixture defect:
observation NPZs overwrite three checkpoint files after the actual restart. The
compiled Program IR is also absent from those archives. Their passes remain
authentic, with these offline proof gaps; no checkpoint image is reconstructed.

Frozen-D fixture@3 and candidate-D fixture@2 retain each checkpoint under a
distinct `*-checkpoint.npz` path. The test pins its bytes immediately after the
native checkpoint call, checks that later observation writes preserve that hash,
and proves the observation/checkpoint paths are disjoint. Both physics branches,
seven Newton controls, finite-difference step and original acceptance thresholds
are unchanged.

The same actual compiled Program component that supplies compiler-owned C++ now
exports its carried IR with `dump_ir()`. This serializes the component's existing
detached Program; it does not re-resolve a builder or infer an IR from C++. The
receipt pins component, IR path/hash and the observed native program hash. Actual
DSO/sidecars and the aggregate association retain their existing distinct proof
scope. Independent readers must receive this new fixture version before any
offline checkpoint/history/IR claim. A fresh native run is required.

There is no production contract, API, SDK or physical-model change in this fixture
repair. The original archives and their raw logs remain immutable.
