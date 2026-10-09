# STATE17 constant-reference counterexample - Source only

Main6b609b5c / CPP99358d38, Coref3e472. No code/probe/run/Main change.

The actual code does not imply Phi=6 and Q=12 on every AMR valid cell after
produce_stage(context,1,2). context.state(program_block) selects active_level_ only
(amr_program_context_spatial.inc:356-362). produce_stage copies/scales that one carrier
(test_prepared_field_rhs_inputs.cpp:372-378). The public Field request authenticates
that active-level override, then explicitly captures accepted STATE on every other
level (amr_program_context_value_authority.inc:93-148, especially130-142). Core selects
these request states per level (amr_system.cpp:7009-7016).

Consequently a sparse two-level composite solve can receive RHS6 on the active coarse
level and RHS3 on the accepted fine level, rather than uniform RHS6. With reaction1,
the screened periodic operator applied to constantPhi6 gives RHS6 everywhere. It
cannot satisfy RHS3 on a nonempty refined region. Q=2Phi is still the actual derived
math, but constantQ12 everywhere is unsupported. This inference does not require
claiming that the genuine STATE source was actually time-integrated by the Program.

Root368a and independent302b receive eight nonfatal constant-reference failures
while source=currentQ exactly passes. Repeated high-reference checks are a plausible attribution from the Source,
but the first helper output is unlabelled; no runtime phase is assigned as established
without the producer's receipt/phase evidence. The negative remains unqualified.

Next authorized bounded correction: retain original16/192 and baselinePhi3/Q6/Psi6
strict checks; snapshot actual accepted highQ into owned immutable per-level Fields
before restore, independently of the source operator under test. Compare source and
currentQ exactly to those numerical references after high capture and high restore;
require exactQ=2Phi and collective max|highQ−6|>existing1e-9 on each global level.
This witnesses storage/binding difference, not a closed mixed-level PDE or convergence.
Add GTest phase/level traces. No tolerance, guard, forcing or accepted fine STATE is
changed to force12. V2 correction3e98d2/3b4bc8 is host-syntax checked and Source-admitted0ac081,
exactly integrated Main2cb88e33. It adds finite-to-infinity checks in only the
two new STATE error kernels. V1 is frozen/unadmitted/unrun due to reviewer869c
NaN masking blocker. First corrected world1 is received PASS1 by independente055 and Rootf45bb;
fullsuite17 receives51 passes in producer32027, independent0cd and Root939.
These exact receipts remain Source2cb/SDK967; later header637c/af8 has no Native
or scientific runtime reception yet. Original negative/invalidhigh12 counterexample remains preserved. Counts17
cases/224 assertions are syntactic Source counts, not runtime executions.

- state_highref_v1_source_blocker (`/Users/romaindespoulain/dev/tmp/PoPS-independent-high-reference-correction-v1-review2-20261008/review.json`), SHA `869c831807d4ceee3b30062726e6305b482a9fccd8bf79a7ac2d80fb4452a88e`.
- state_highref_v2_source_admission (`/Users/romaindespoulain/dev/tmp/PoPS-independent-high-reference-correction-v2-review2-20261008/review.json`), SHA `0ac0815fc0c91c8f3670a3c32de35cc768d7bbf9269f3233c36593373feeb1f4`.

Current runtime links: first v2 independent (`/Users/romaindespoulain/dev/tmp/PoPS-independent-accepted-state-witness3-v2-world1-reception2-20261008/reception.json`), first v2 Root (`/Users/romaindespoulain/dev/tmp/root-accepted-state-witness3-v2-world1-positive-2cb88e33-20261008.json`), suite17 producer (`/Users/romaindespoulain/dev/tmp/PoPS-accepted-state-witness3-v2-CPP-suite17-2cb88e33-20261008/complete-receipt.json`).

Suite17 closed refs: independent0cd (`/Users/romaindespoulain/dev/tmp/PoPS-independent-accepted-state-witness3-v2-suite17-reception2-20261008/reception.json`), Root939 (`/Users/romaindespoulain/dev/tmp/root-accepted-state-witness3-v2-suite17-positive-2cb88e33-20261008.json`). Source Main637c/af8 integration does not inherit these SDK967 receipts.
