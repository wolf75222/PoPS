# T3 local products: implementation and bounded evidence

Base: `adedca04f3b81125ff1854ce9e2f26aa5af6c518`, isolated
`PoPS-resource-lifetime`, branch `codex/api040-t3-products`. No MAIN edits, package
installation, heavy native build or MPI/device campaign.

Red: the public `LocalResidual(..., initial={"a": a, "b": b}, captures=...)`
2+3 system failed at authoring with `initial_guess must be a State value`.
Both initial source probes failed before implementation. The full square system
has an invertible Jacobian even though its leading 2x2 diagonal block is zero;
forcing block elimination would be mathematically wrong.

Implemented: canonical named product packing; independently qualified frozen
captures; immutable callback unknowns; exact residual keys/widths; hidden capture
refusal; per-State support/position retention. The product contributes a closed
original-residual DAG to the existing coupled kernel/provider, not another solver.
Common Program checked expression lowering preserves lazy branches and domain
checks. Every original equation is evaluated by the prepared nonlinear provider.
Collective exact layout/distribution/local-rank validation precedes `fab(li)`.
Candidate scratches are projected only after collective outcome consumption.
The layout guard also protects the prior coupled implicit Euler shell.

Executed on this source:

- 63 targeted tests passed: product source and emitted host provider, existing
  coupled emitter, scalar expression/adversarial/authoring-atomicity suites, and
  original-residual stagnation tests. Host fragments use the real
  `prepared_local_nonlinear.hpp` at C++20/O2 with `-fno-fast-math`.
- Public Case validation, resolution, detached ProgramModelGraph and System
  emission pass with two distinct Models, heterogeneous widths and block order
  permutation. A real cross-unknown monomial is included in the public case.
- Host solves verify original residuals for an affine system with singular diagonal
  block, a cross-unknown nonlinear monomial, changed seeds, re-bound captures,
  an unsolvable last row, an invalid sqrt masked by minimum, and lazy where.
- Foreign same-name Program captures refuse without changing authoring identity.
- Four installed integration variants collect successfully; **not executed**.
  They exercise public bind/run, two capture rebinds, block permutation and complete
  state/time rollback on a failed five-equation solve. Root must execute serial
  and MPI2 against a freshly rebuilt artifact.
- Ruff and diff whitespace checks pass.

Exact targeted command:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q -o pythonpath=python \
 tests/python/unit/codegen/test_local_residual_product.py \
 tests/python/unit/codegen/test_local_residual_product_host.py \
 tests/python/unit/codegen/test_coupled_implicit_codegen.py \
 tests/python/unit/time/test_program_expressions.py \
 tests/python/unit/time/test_program_expressions_adversarial.py \
 tests/python/unit/time/test_program_authoring_atomicity.py \
 tests/python/unit/codegen/test_api040_t3_original_residual_stagnation.py
```

Expanded exploratory suite: 81 passed, 3 skipped, 6 native-manifest setup errors,
4 source failures. The six tests require a native variant unavailable in this
source-only checkout and never compiled a new artifact. The four old
`test_time_ops_polish` tests invoke RHS emission without a ProviderPack; all four
were reproduced against the unchanged `adedca0` package in an isolated temporary
source copy. Evidence: `outputs/t3-products-baseline-existing.log`. They were not
patched or counted as passed.

Remaining implementation obligations: `P.source`/`P.apply` nodes inside the new
product callback, general global-product SolveRequest realization, rectangular
methods, elimination/reconstruction contracts, joint nullspaces/compatibility,
and native MPI/GPU/AMR qualification. No completion claim for all T3/C16–C19.
The versioned API contract is `docs/development/api_040/local_residual_products_v1.md`.
