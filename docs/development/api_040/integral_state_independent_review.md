# Independent IntegralState review

Reviewer: GPT-6 Astra/high, independent of the implementation author. Test commit: `5a99502`.
Implementation revision: `88eba605f86601f1a7f15098f38109f5be821e1d`. The tracked author checkout was clean on replay. The five source tests passed in 3.43 seconds with `pops-api040-c11` Python; the independent host harness compiled with C++20/O2 and passed. These are source/host checks, not an installed-package receipt.

## Verified seams

Five source tests exercise a public conservative FV case with a two-stage SSPRK update, no diffusion. The two evaluations of one physical occurrence select different authored evaluation identities. An unaccepted diagnostic rate cannot deliver a trace, and a foreign Program cannot reuse the handle. Detachment preserves the semantic serialization, integral identity, selected rate IDs and emitted code. Declaration is emitted before installation of the step callback. The AMR target emits one parent-only delivery in its post-synchronization phase, dispatched after hierarchy advancement.

The independent C++20/O2 harness compiles the real `accepted_exchange.hpp`. It checks exact external axis/side/component/occurrence selection, exclusion of internal incidences, persistent initial value 0.7, refusal of duplicate consumption without mutation, overflow refusal without consuming the input, a successful retry, a fresh exchange with a different temporal weight, and POPSEX01/POPSEX02 roundtrips. Two SSPRK evaluations with weights 0.05 and fluxes 2 and 4 deliver 0.1 and 0.2 separately; restart between deliveries preserves the second delivery and refuses replay of the first. Signed-zero initial declarations remain distinct. A forged standalone source-evaluation field conflicting with its qualified runtime frame is refused during checkpoint import.

## Findings and repairs examined

- The original selector identified only the physical operation and occurrence, so two stage evaluations could be consumed together. The source evaluation identity now remains distinct from the runtime dt/clock qualification and participates in selection and wire serialization.
- The public conservative-only source test initially failed: the transport-retention gate required an independent diffusion construction, leaving no captured face carrier. Explicit integral consumers now request retention of their exact FV operation without adding a physical diffusion or a fictitious flux. This was a reproduced source failure, followed by a passing source test.
- Per-level AMR delivery would reject a coarse level whose selected physical boundary was entirely covered by fine cells. Delivery was moved after hierarchy advancement/reflux, guarded to the parent once, while retaining the collective refusal of a genuinely absent external trace. This repair has source evidence only.
- The replicated scalar value must agree across MPI ranks at declaration, delivery and restore, even though the distributed face records legitimately differ. Exact collective value-contract checks are present at these boundaries; MPI execution remains pending.
- Full integral identities, rather than short author names alone, are included in the checkpoint capacity calculation. The extra source-evaluation text and consumed-key storage are also counted. This is a source review, not an exhaustive capacity proof.
- A suspected initial-value issue was ruled out: the generated prelude executes during installation before the step callback; it is not postponed to the first time step.

## Authority and version boundaries

On the frozen author revision, Public API 2→3 exposes persistent integral authoring/reading; Semantic IR 2→3 and Program serialization 4→5 include their declarations and exact transfers; native release ABI 3→4 and generated System package ABI 6→7 require rebuilding providers calling the new native methods. The ledger uses POPSEX02 only for integral-bearing plans and preserves POPSEX01 for plans without declarations. These boundaries are explicit; they are not an artifact-snapshot schema migration.

Module manifest 10, artifact snapshot 6, AND9, Uniform checkpoint payload 8 and Python AMR checkpoint payload 11 remain distinct and unchanged. The compiled package identity includes the package ABI, and the release/SDK contract must match at reception. No claim of general checkpoint compatibility follows from the ledger's legacy POPSEX01 roundtrip. Dedicated loader tests rejecting a stale package remain part of native reception, not a consequence of this host harness.

## Required native reception

No remaining concrete source defect was found in the reviewed seams after these repairs. These tests do not establish PDE correctness, complete W11, a sheath/M14 model, MPI execution, AMR composite delivery, or a GPU backend. The native reception must use the rebuilt package and include q immediately after bind; actual retained external faces and nonperiodic boundary orientation; two temporal stages; an MPI rank owning no selected face; collective absent-support refusal; provisional child acceptance followed by parent rejection; retry with a different dt; checkpoint into a fresh runtime; and an AMR selected boundary fully covered by fine cells. Quantity and consumed-key restoration must both be checked. The author's public native fixture now covers Uniform, AMR1 and AMR2 with a coarse boundary entirely fine-owned, but those fixtures were only collected by the author and were not run in this review. The C++ native transaction fixture similarly remains pending. No threshold or physical equation was changed by this review.

## Reproduction

From a checkout containing the frozen implementation and the independent tests:

```sh
rtk proxy env -u PYTHONPATH /path/to/pops-env/bin/python -m pytest -q --tb=short -o pythonpath=python tests/python/unit/codegen/test_integral_state_independent.py
rtk proxy clang++ -std=c++20 -O2 -I include tests/cpp/unit/runtime/integral_state_independent_host.cpp -o /tmp/integral-state-independent-host
rtk proxy /tmp/integral-state-independent-host
```

The host command uses no MPI compile definition and no installed native extension. It therefore cannot substitute for the native transaction/MPI test.
