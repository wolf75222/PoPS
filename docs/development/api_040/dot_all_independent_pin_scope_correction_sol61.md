# Independent dot-all receipt: scope of the frozen source pin

The original independent test pinned the entire Uniform ProgramContext header to
`9471bf8ebceda549e75355ed0b9f2d77133d8a06`. The integrator's SDK7b source reception
reported one test failure when T5 added unrelated candidate capture methods to
that header. That comparison exceeded the reviewed reduction's dependency scope;
it was not evidence of a changed dot-all body.

The corrected probe still authenticates against **9471bf8**, not against the new
header. It compares the exact Uniform and AMR `Real dot_all(int program_block, ...)`
bodies, their collective error helpers, Uniform field-contract/activity lookup,
and AMR active/finest-owner traversal bodies. The actual finite product kernel,
compensated accumulator, Kokkos reducer and collective two-word helper remain
covered by the unchanged full-file pins for `mf_arith.hpp` and `for_each.hpp`.
Assertions and the 105 arithmetic/transport checks are unchanged.

`POPS_REVIEW_SOURCE_ROOT` optionally selects a read-only source checkout; the
trusted production pin remains hard-coded. With the current MAIN source (observed
HEAD `163ab41bfb9026033d320cf167cdcffd8e8f7d98`) selected this way, **both independent
source/host tests pass**. MAIN was only read. No installed PoPS package, native
Program, MPI run, build or JIT was used. T5 offline work is in a separate tree.

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM POPS_REVIEW_SOURCE_ROOT=/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/work/PoPS /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I -c 'import sys; from pathlib import Path; sys.path.insert(0,str(Path.cwd())); import pytest; raise SystemExit(pytest.main(["-q","--tb=short","tests/review/test_sol61_dot_all_compensated_independent.py"]))'
```

This corrects the historical receipt's whole-header drift assertion. Native
results reported by the integrator remain distinct from this source/host replay.
