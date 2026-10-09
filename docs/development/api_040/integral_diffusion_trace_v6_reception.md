# Prepared constitutive external-trace extension, Program IR 6

Base reviewed production: `64b9e15046c901dfa1a8d83adbf4466ba35f230c`.
Independent coverage review/tests are a separate commit, `15c1dc4`.
This extension is implementation work, requiring an independent production review
and rebuilt native reception by root; it is not an independent approval of itself.

## Previous public boundary and exact extension

The author previously required `rate.op == "rhs"`, rejecting `diffusive_rhs` at
`accept_external_trace`. The transfer emitter also captured only retained FV
transport. There was therefore no silently accepted diffusive scalar update.
Prepared diffusion already published accepted conservative face incidences, but
left `axis/side/component/exterior_trace` unset. The extension allows the public
diffusive Rate and binds it to exactly the operation and occurrence already staged
by `_emit_diffusive_accepted`, plus its exact source evaluation identity.

Accepted diffusion quadrature is recomputed with the existing affine/SSP or
partition-stability proof and the current availability of its evaluated carrier.
A diagnostic evaluation outside the accepted commit cannot supply the scalar.
One exact constitutive occurrence is admitted, including the inseparable fitted
drift/diffusion carrier. Independent transport+diffusion occurrences in the same
Rate remain ambiguous without an occurrence discriminator and are explicitly
refused. Sources are excluded from selection.

The generic C++ producer writes axis, side, component and the context's global
physical-boundary decision into each existing record. It retains actual numerical
flux, measure, orientation, coefficient/quadrature weight, occurrence and evaluation;
no equation, face evaluation, reflux, geometry or time-step formula changes.
Its EB and finest-owner predicates remain intersected before face enumeration.
The context and prepared lookup remain the authority for physical topology and
global AMR owner coverage. Uniform and AMR use the same record schema and consumer;
AMR delivery stays once after hierarchy synchronization.

Program serialization is conditionally version 6 for diffusive selectors. Existing
transport-only programs stay version 5; existing transport tests assert their
identities and version. POPSEX02, native integral methods and record fields already
support the added producer and are unchanged. Header/SDK authentication still
requires rebuilding before native qualification.

## Source and host evidence

The public fixture uses `u=1+x`, `D=.1`, x value traces `1+x`, periodic y, `dt=1e-4`,
`q_right(0)=.7`, `q_left(0)=-.2`. The independent oracle gives opposite accepted
amounts `+.1*dt` and `-.1*dt`; field mass and their sum stay constant. AMR2 tags
the right half, leaves the left physical trace coarse-owned and the right trace
fine-owned, with fine substep weight `dt/2`. No expected amount comes from a native
decoder or the generated diffusion operator.

Source tests resolve/emit Uniform and AMR2 public Programs; verify exact operation,
occurrence/evaluation selectors, conditional version and delivery scope; reject
unaccepted diagnostics and ambiguous occurrences. The host consumption test
extends only the existing harness main, retaining the exact extracted production
`stage_accepted_exchanges` method and production `AcceptedExchangeLedger` header.
It authenticates two fine runtime frames, excludes covered coarse faces, selects
16 actual records, updates `.7` to `.712` for the host's explicit flux density 1.2,
rejects double consumption, preserves consumed keys across POPSEX02 restore, and
replays after restoration of the initial image. This is a host ledger/control-flow
seam, not a native AMR preparation, diffusion stencil or transaction execution.
Exact pre-extension header `git show 64b9e150` fails that same host test with
`diffusion trace metadata is absent from exact selector`.

## Native reception prepared, not run by this worker

`tests/python/integration/runtime/test_integral_diffusion_trace_runtime.py` has four
collected tests: Uniform and subcycled AMR2 balance/checkpoint/restart, absent periodic
trace rollback, and unsafe-dt rollback followed by safe endpoint-clipped retry.
They compile the public fixture and save authentic state/checkpoint/wire receipts;
the independent POPSEX02 reader verifies selected keys, orientations, component,
flux density, measure, level/substep weights, consumed sets and scalar values.
The tests use the existing MPI collective-call/compile-once/check helpers and can
run unchanged under installed MPI. Two C++ unit contexts and the MPI3 context now
provide physical-trace classification and source-evaluation qualification explicitly.
Their physical diffusion/MPI3 tests also assert exterior metadata and exact trace
selection. Root must compile and execute those C++ and Python tests against the
matching installed SDK. No native JIT, environment mutation or heavy build was
performed here. Active EB physics remains explicitly unsupported by the existing
diffusion realization; ledger mask intersection is separately host-tested, not a
claim of native EB diffusion support.

The installed-import SDK discovery repair for the recovered coverage harness is
owned by root and must be integrated with these tests; this worker did not edit
that harness after its independent-review commit.

Commands (from this exclusive checkout, source authority explicitly selected):

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -c 'import sys; sys.path.insert(0,"python"); import pops,pytest; print("SOURCE",pops.__file__); sys.exit(pytest.main(["-q","tests/python/unit/codegen/test_integral_diffusion_trace.py","tests/python/unit/codegen/test_accepted_exchange_coverage_independent.py","tests/python/unit/codegen/test_integral_state_trace.py","tests/python/unit/codegen/test_integral_state_independent.py","tests/python/unit/codegen/test_diffusion_program.py","tests/python/unit/codegen/test_diffusion_frozen_input_quadrature.py","tests/python/unit/codegen/test_integral_state_public_receipts.py"]))'
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -c 'import sys; sys.path.insert(0,"python"); import pytest; sys.exit(pytest.main(["-q","--collect-only","tests/python/integration/runtime/test_integral_diffusion_trace_runtime.py"]))'
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check python/pops/codegen/program_integral_transfers.py python/pops/codegen/program_emit_diffusion.py python/pops/time/_program/integrals.py python/pops/time/_program/serialization.py tests/python/support/integral_diffusion_case.py tests/python/unit/codegen/test_integral_diffusion_trace.py tests/python/integration/runtime/test_integral_diffusion_trace_runtime.py
```

Local source/host checks are not GitHub CI, MPI runtime evidence or native AMR
qualification. No full W11 circuit, M14 sheath or general implicit diffusive Rate
selector extension is claimed by this bounded change.

Results: **36/36 source/host tests passed in 26.84 s**, four native tests collected
in 0.53 s, Ruff passed, and `git diff --check` passed. Compiler for the host ledger
seam: Apple clang 21.0.0 (`clang-2100.1.1.101`), C++20, `-O2`, arm64 Darwin.
The earlier narrower source/host run was 28/28 in 19.53 s. Native test execution
and exact rebuilt installed-package reception remain outstanding for root.
