# Independent review: accepted AMR trace coverage

Reviewed integrated production: `64b9e15046c901dfa1a8d83adbf4466ba35f230c`,
branch `codex/api040-sol61-coverage-review`. Recovered the earlier independent
tests and receipt byte-for-byte from `PoPS-principal-group`, then rechecked source,
historical wire data and executable controls in this exclusive checkout.
The earlier coverage producer was `755b9096e11498599cfaf427e82a3ce1a3fa59e3`;
the local-lookup follow-up was `2195ed6b520ad5e01eb03f2230a297db386588a4`.
This receipt covers source inspection and small host C++20/O2 tests. It does not
claim an installed AMR/MPI run, the full W11 circuit, or M14 sheath qualification.

## Actual installed failure, decoded independently

The original root receipt is preserved in workspace
`outputs/integral-amr-duration-probe/`. Its `summary.json` identifies source
`7443b8301344ee46369cdb4c7fb77cab48e9729f`, installed Dim2, MPI enabled,
SDK `b7b9b5bf115b91275e4edc0d1e600df5b34fbd01879ff1ec826b2bd20ad1de86`.
The independent POPSEX02 reader in `tests/python/support/integral_state_receipts.py`
was rerun against `program-exchanges.bin` and consumed its bytes completely,
without using the native decoder. This is historical installed evidence from
`7443b830`, not evidence of a rebuilt current native extension.
Wire SHA256: `0e8ca1b9c71f7c445feff994fdd6a90cca9836d2682e8d64cfb69c75d4bdf94a`.
It contains 23,552 records and 160 consumed keys; every selected exterior x+
record belongs to the consumed set.

| Level/substep | Exterior x+ records | Total measure | Temporal weight | Signed amount |
| --- | ---: | ---: | ---: | ---: |
| 0/0 | 32 | 1 | .01 | -.012 |
| 1/0 | 64 | 1 | .005 | -.006 |
| 1/1 | 64 | 1 | .005 | -.006 |

The initial quantity is .7, the saved quantity .724. The frozen oracle is .712.
Fine-level durations are correct. The extra contribution is the covered coarse
boundary. Root's saved coarse mask has no active x+ cell; this review independently
confirmed the ledger values and keys, rather than rerunning the simulation.
The saved `state.npz` coarse mask has shape `(32,32)` and zero nonzero entries
in its rightmost column, independently rechecked during this review.

## Mechanism and findings

`pointwise_active_mask` selects embedded-boundary activity only; without an embedded
boundary it returns nullptr. It never certified finest-owner AMR coverage. The new
`pointwise_exchange_coverage_mask` selects `PreparedHierarchy::active_coverage`,
whose construction clears the coarse footprint of finer patches. The prepared
lookup authenticates exact level, layout, distribution, local rank and patch count.
The producer intersects both masks before emitting cell incidences. This changes
accepted-exchange enumeration, leaving the native face field, reflux, local dt,
physical boundary selection, orientations and temporal quadrature unchanged.
Uniform returns no coverage mask. Both retained transport and prepared diffusion
use the same two-mask contract.

Two additional findings were reported and repaired before final source approval:

1. The initial coverage accessor called `refresh_resources_` after the existing
   active-mask accessor. The latter can reject a layout locally after its collective
   preparation; other ranks would then enter a second `ensure_engine` Allreduce
   before the producer's error vote. The follow-up makes the second lookup strictly
   local against the already-prepared hierarchy. It keeps layout validation.
2. Three native test contexts lacked the new method: two in
   `test_prepared_diffusion.cpp` and one in `test_mpi_exchange_batches.cpp`. Their
   Uniform adapters now explicitly return nullptr. No silent production fallback
   was introduced.

## Independent executable counter-tests

`tests/python/unit/codegen/test_accepted_exchange_coverage_independent.py` compiles
the actual transport emitter with the production `ExchangeRecord` header and a
small explicit geometry/field adapter. It checks fully fine-owned x+, an interface
intersecting x+ (half coarse/half fine), EB/coverage intersection, two components,
an empty local rank, unchanged Uniform enumeration, and foreign-mask layout refusal
before any batch publication. The pre-fix emitter fails with
`covered cell published a trace record`; exact source `git show 7443b830` was
compiled again with the same scaffold and fails, while the integrated emitter passes.

A third host test extracts and compiles the actual
`PreparedDiffusion::stage_accepted_exchanges` method. It supplies already evaluated
flux densities and exercises the physical-boundary-only path, including periodic
face exclusion. It checks the same fully fine-owned/mixed/EB/two-component/empty/
Uniform/foreign-layout cases. A coverage-bypass mutation of that exact method
fails with `covered cell published a diffusive record`. The only source adaptation
is its `Index<Dim>` spelling to the scaffold's two-dimensional index type.
Diffusion's records retain incidence identities and do not populate transport's
`exterior_trace` metadata; these checks qualify enumeration and signed ledger
amounts, not a diffusive IntegralState transfer or a native face evaluation.

A second host test extracts and compiles the actual AMR coverage accessor. Any
attempt to call its second collective preparation raises a test exception. Exact
`git show 755b9096` fails with `coverage lookup reentered collective preparation`;
the local-lookup follow-up passes, including the foreign-layout refusal.
This is a control-flow seam, not an MPI deadlock execution or an AMR preparation test.

The integrated transport/diffusion/accessor tests plus five independent IntegralState
source tests passed **8/8 in 14.14 s**. Ruff passed. Compiler: Apple clang
21.0.0 (`clang-2100.1.1.101`), host `arm64-apple-darwin25.5.0`, C++20, `-O2`.
The source-only import printed this checkout's `python/pops/__init__.py`; the shared
installed package was not used as the Python source authority.

From this checkout, the exact reception commands were:

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -c 'import sys; sys.path.insert(0,"python"); import pops,pytest; print("SOURCE",pops.__file__); sys.exit(pytest.main(["-q","tests/python/unit/codegen/test_accepted_exchange_coverage_independent.py","tests/python/unit/codegen/test_integral_state_independent.py"]))'
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check tests/python/unit/codegen/test_accepted_exchange_coverage_independent.py
```

The negative controls replace only the Python emitter with `git show 7443b830`,
the accessor header reader with `git show 755b9096`, or remove the diffusion
coverage predicate in memory. Each invokes the same test entry point in a separate
temporary directory and requires its exact failure diagnostic. No production file
or shared installation was mutated by these controls.

No native JIT, shared installation or heavy build was executed by this reviewer.
These are local source/host results, not GitHub CI or native MPI/AMR qualification.
Root must receive the rebuilt AMR2 case and MPI
case, preserving q=.712, the wholly fine-owned boundary, and the per-substep ledger
assertions. Public restart/SSPRK2/rollback receipts remain separate tests.
In particular the real three-rank, rank-local layout-mismatch test in
`test_mpi_exchange_batches.cpp` belongs to root's native compilation/execution.

The earlier pre-build review also missed the release ABI4/header kAbiVersion3
mismatch subsequently caught and repaired by root's integrated build. That failure
is retained as a limit of source/host evidence, not rewritten as a prior success.
