# Independent reception of explicit vector pairing 21a56b9

Date: 30 September 2026. Exact candidate:
`21a56b910c421eee465afcb6f16123dd309360a9`, parent
`8bf5ae0c3e028aefcf0b57f70a195017a896edfd`.
Exclusive checkout `work/PoPS-sol61-dot-all-review`, branch
`codex/api040-sol61-dot-all-review`. No production, installed package or SDK change
was made by this review. Native execution belongs to the central reception.

## Blocking finding: AMR coverage is omitted

The new contract and its documentation promise an all-component sum over finest
owned active cells. The actual AMR `dot_all` calls
`for_each_owner_active_level_`, whose state/history route visits every hierarchy
level. That visitor only asks for `prepared_amr_block_level_active_mask`; it never
queries `prepared_amr_block_level_coverage_mask`. The real active getter returns
null in Cartesian mode and returns only embedded-boundary activity otherwise
(`src/runtime/amr/amr_system.cpp:13031`). The separate coverage getter is at 13013.
`prepare_active_coverage` initializes ones and writes zero into covered coarse
footprints (5361–5405). There is no implicit coverage combination in the active getter.

The independent host probe extracts the **complete unchanged visitor**, both actual
new provider methods and their actual field-contract validators from the immutable
candidate. Two state levels each containing (2,3), with coarse coverage 0 and fine
coverage 1, return **26**, while the contract requires **13**. The coverage getter
is queried zero times. Explicit EB masks of 1 on both levels retain 26; an explicitly
combined coarse/fine mask 0/1 gives the correct 13. This reproduces the numerical
consequence and confirms that coverage authority must be consumed. Preserve legacy
`dot`/norm behavior; fix the new path with a specific authenticated coverage consumer.

## Nonfinite results need an explicit new-contract policy

The actual Uniform provider also returns infinity after summing finite component
products `1e308 + 1e308`, infinity after a simulated cross-rank sum of two finite
`1e308` values, and NaN from an active poisoned tail. The method catches exceptions,
but floating-point overflow/NaN does not throw. It neither checks the local scalar
before its error vote nor the global scalar after the sum. These observations are
recorded separately from the coverage defect. A strict finite new pairing needs
local nonfinite convergence before numeric reduction and global-result validation.
Legacy component-zero operations must retain their existing meanings.

## Received source and host properties

Thirteen new independent source tests pass in 1.69 s. They receive:

* exact URI `pops.program.dot-all@1` and attrs;
* conditional v5/v7 selection for flat, branch, nested branch, while condition and
  dt-bound regions, without changing ordinary legacy programs;
* actual emission of `ctx.dot_all` in the flat/control-flow regions, alongside
  unchanged `ctx.dot`;
* missing/wrong URI and unknown reduction refusal by the real emitter;
* invalid types, cross-block operands, mismatched known widths, absent block owner,
  foreign Program and forged reversed component Space refusal before publication;
* complete legacy serialization and actual Program hash equality against a separately
  imported immutable parent Python archive, including dot/norm2/norm_inf and a commit.

The dt-bound test receives recursive **serialization only**. An additional direct
dt-bound emission experiment with a readonly block fails the existing block-index
map seam, also relevant to legacy bounds. Reusing an already-authored top-level
state instead reaches the old isolated-bound SSA variable seam. These failures
were inspected rather than treated as successful emission. The separate central
readonly-input fix must be received in the integrated candidate. The author tree
has no central constitutive v6 selection; preservation of that profile and generated
v7 capability metadata also require integrated reception.

The unchanged affected-area source selection passed **155 tests in 15.82 s**,
including twelve of the new tests. Adding the final foreign-Program/reversed-Space
probe produced the final thirteen-test result above. The thirteen independently
frozen vector witnesses from 7ac0b26 pass on the candidate too. Their exact values
remain vector 208 under rotation, component-zero 4 to 9, corrupted tail 20404 versus
legacy 4, and owned raw/comp0/physical values 104/39/29.25. Declared Gram pairing
remains distinct from raw Euclidean coordinates. These are scientific discriminants,
not a claim that arbitrary mixed physical units define one universal norm.

The host probe receives **21 checks**: all-component rotation/tail discrimination,
exact width/layout/distribution/rank/ghost rejection in Uniform, local kernel failure
convergence before numeric sum, collectively refused mask preparation, inactive NaN
exclusion, overflow/NaN observations, AMR refresh/owner refusal, scratch one-level
selection, empty-rank participation and the coverage counterexample. Its storage,
local dot kernel, owner classifier, prepared lane and serial collective implementations
are substitutes. The actual provider methods, validators and complete level visitor
are unchanged. This qualifies their extracted control flow, not native MultiFab,
AMR hierarchy preparation, ownership classification or MPI collectives.

## Collective preflight assessment

The Uniform active-mask call occurs before `dot_all`'s local try. The real call chain
already converges ordinary errors: `resolve_pointwise_program_block_` catches and
votes before route consensus; `System::prepared_program_block_active_mask_` catches
field/mask layout errors and votes before EB mode/generation consensus. Therefore
the position alone is **not** evidence of a rank-local failure bypass for those
ordinary inputs. It also means the full call performs preflight collectives in
addition to the advertised reduction error vote and numeric sum.

Both providers acquire the prepared lane first. AMR additionally refreshes prepared
resources inside its visitor; that path itself enters collective hierarchy refresh.
The host substitutes do not receive corrupt prepared-context/lifecycle failures
across ranks, so no stronger claim is made about such failures. Real MPI fault
injection must cover invalid layout/owner, empty ranks and coverage preparation on
the corrected exact SDK. The new local numerical checks must precede its numeric
collective; post-sum overflow must also fail explicitly.

## Reproduction and integrity

```sh
rtk proxy /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_dot_all_host_21a56b9.py
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/review/test_sol61_dot_all_21a56b9.py
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/review/test_sol61_vector_reduction_oracles.py
```

The last file is already frozen in independent review commit 7ac0b26 and was copied
unchanged for reception, not duplicated in this review commit. The raw host receipt
is `outputs/sol61-dot-all-21a56b9-host/receipt.json`; it records each extracted function
hash and the compiler `/usr/bin/clang++`. Exact header/source archive SHA256:
`777749c7f06dfdffa3707d74db35ca21bc42022ea72b97f9a44c3baca7ad7ae5`.
Host scaffold SHA256:
`3d7803c93399bfb727f781273fb613bf3b62a98ee007725f46b765f268da15b8`.

An attempted author runtime unit-file collection was blocked by its eager native
bootstrap import with no native dimension selected; deselecting native test names
does not bypass collection. It is not counted as passing. The independent source
and host checks require no installation, environment mutation or heavy JIT.
Ruff and diff-check pass. Candidate 21a56b9 is **not received for global integration**
until coverage and numerical failure policy are corrected and replayed. No native
rotation/retry/restart, MPI/GPU, PDE or physical-norm qualification is claimed.
