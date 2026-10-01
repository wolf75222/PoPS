# Independent Stage fixture archive reception

Candidate: `c3802111e9f0151c7b9ab43e2fd629c967a56a60`, parent
`be71d6c64ddc9c0b7048250863fb39f374293397`. This review changes tests and
documentation only. It does not execute PoPS, import its package, load a DSO,
compile, or construct native checkpoint evidence.

The fixture calls the real runtime checkpoint API at three distinct targets:
`accepted-checkpoint`, `continuous-checkpoint`, `replay-checkpoint`. Each
returned file is hashed immediately, before its corresponding history/state
capture. The later observation files retain their separate `accepted.npz`,
`continuous.npz`, and `replay.npz` names. Final rehashes must equal those
immediate hashes; resolved checkpoint paths must be disjoint from all five
observation paths, including initial data.

Thirteen independent host/protocol checks pass. They execute the complete
final guard extracted unchanged from the actual fixture, with explicitly
synthetic bytes. All three overwritten checkpoint cases refuse with
`native checkpoint was overwritten`. All three observation collisions, both
direct and through symlinks, refuse with
`checkpoints and observations must use distinct files`, even when the
immediate test pin is resealed to reach that second guard. The positive case
retains three checkpoints and five observations. No synthetic file is a
native NPZ or proof of checkpoint physics.

The source check also verifies all three concrete target names, seal-before-
capture ordering, and fixture schema `pops.evolved-stage-native-fixture@2`.
The export loop obtains Program components from `artifact.layout_programs`
and exports C++, IR and `program_hash` through the same `component` variable.
The actual `CompiledProblemDumpMixin.dump_ir` implementation serializes its
carried Program; executing that unchanged method with a synthetic carried
payload verifies exact JSON output and refusal when the Program is absent.
The real loader computes `program_hash` from its detached carried Program at
construction (`python/pops/codegen/loader.py`). These are source/protocol
checks, not an attestation that a particular DSO was compiled from the dump.

Run from the private checkout:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONDONTWRITEBYTECODE=1 \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q \
  tests/review/test_sol61_stage_archive_independent.py
```

Result: **13 passed**. No new blocker is demonstrated in this bounded review.
The final path guard authenticates checkpoint/observation separation; it is
not a general closed archive or external owner-seal verifier. It does not
independently reject two returned checkpoint paths aliasing one another when
their bytes and pins coincide; the three actual API targets are distinct,
and authentic runtime return paths remain an obligation of native reception.
Fixture@1 retains its eight historical generated-C++ failures and cannot be
upcast to a positive @2 receipt. Native Stage science, exact restart/replay,
MPI, AMR conservation and compiler/DSO provenance require fresh ROOT reception.
