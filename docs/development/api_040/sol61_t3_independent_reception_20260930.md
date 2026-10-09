# Independent T3 source/math reception

Date: 2026-09-30. Reviewer: GPT-6.1 Sol. Implementation received:
`cda144e91704bc8da7c65f1c7cc4bcc9b610391c`, parent
`2599d2214bc34bbb351dec2bda63bcbab381bf89`. The independent refusal counter-case
was then replayed against its bounded author correction
`008bd383a59eb0afc9fe075acb1c54dc0bd93a60`.

The exclusive `PoPS-sol61-spatial-nonlinear-review` checkout contains these exact
sources. No production edit, header, principal checkout, environment install,
JIT, or native Program execution is part of this review.

## Result and concrete corrected defect

Source/math reception is positive after `008bd383`: **97 tests pass**, including
21 independent tests. The original coupled residual, exact capture routing,
seed distinction, five-unknown product, full-residual derivative selection,
registered source authority, and explicit Uniform scope have independent probes.
A full generated five-unknown/two-capture/two-solve C++ translation unit passes
syntax checking against the real Dim2/Kokkos/MPI headers.

On the original `cda144e9`, swapping `capture_0` and `capture_1` in an otherwise
valid public `SolveRequest` raises `SymbolicTruthValueError` rather than the
declared `SolveRequestError [equation_input_mismatch]`. The new binder compared
dicts containing symbolic `ProgramValue`s: unequal values invoke symbolic
equality and cannot have a Python truth value. The failure already preserves
the serialized Program and node/region counters exactly; there was no accepted
publication. The author corrected it in `008bd383` by comparing exact keys and
each SSA binding with `is`. The same independent public attack now receives
the typed refusal and exact authoring rollback. No tolerance or valid graph was
changed by that fix.

## Independent original-equation witness

The mathematical oracle from `576b974` is retained byte-for-byte in
`tests/review/test_sol61_spatial_nonlinear_math_oracle.py`. The new
`tests/review/test_sol61_spatial_field_cda_independent.py` authors its own public
FieldProblem rather than reusing the implementation author's case builder.

The witness uses 2, 3, and **5** co-located scalar unknowns, a periodic anisotropic
5-by-4 grid on lengths 1 and 1.7, two distinct captured State blocks, cubic
reactions, products of different unknowns, and a nonsymmetric cross-diffusion
matrix. Every equation has the original form

```
R_i(u) = α_i u_i + β_i u_i³ + A_i(x) u_i Σ_j C_ij u_j
         + Σ_j B_ij u_j² − Σ_j D_ij Δ_h u_j − f_i(x).
```

An independent interpreter of the candidate's authenticated closed AST plus its
encoded diffusion matrix agrees with this original oracle at the manufactured
solution and at two distinct off-solution seeds. It receives canonical,
three-component and five-component permutations. The oracle's dense analytical
Newton independently converges from two seeds and checks every original
equation; it is not the candidate's native Newton/GMRES execution.

The generic five-unknown public Program emits two solves with the same equation
identity and different initialization identities. Each has its own output and
capture buffers; observations explicitly route each original unknown to its
history. Arbitrary renaming emits the same numerical operation family. The
generated code invokes the real `PreparedSpatialResidual`, its full spatial
`apply_general_field<..., 5, 25>`, and original final recheck, with no reaction
inverse or condensation branch. These are source/emission facts, not saved
native solved values.

The prepared native derivative uses central differences of the complete
residual, including diffusion:

```
s = selected_step * max(1, ||q||₂) / ||v||₂
Jv ≈ [F(q+s v) − F(q−s v)] / (2s).
```

The selected base step is explicitly `1e-7`. The independent scaled central
action agrees with the analytical Jacobian of the original five-field relation.
Other oracle probes distinguish forward differences, reaction frozen at the
seed, transposed diffusion, misplaced captures/RHS, and a tail-component defect.
Neither this selected unweighted Euclidean norm nor its FD normalization is
claimed as a universal norm or scale for heterogeneous physical units.

## Authority, refusals and collective source inspection

Independent public attacks reject swapped captures, a foreign-Program seed,
invalid output identities and an unsupported exact derivative route. The
serialized Program and node/region counters remain unchanged. An unsupported
nonhomogeneous Neumann boundary fails before an accepted Program is formed.
The explicit AMR target is rejected; no AMR nonlinear realization is implied.

A coherently resealed FD policy still fails against the registered numerical
method. An annotated literal unit fails explicitly without a unit-system
conversion before lowering. This is unit admission, not a silent conversion or
qualification of arbitrary mixed-dimensional residual coordinates. The
registered physical equations are recompiled at lowering; internally resealing
node metadata does not create new physical authority.

The generated input-layout preflight votes before copying captures or using a
seed. The native general-field operator catches storage/boundary mismatches and
votes before halo fill; in particular it checks each declared periodic/Neumann
face against the prepared topology. The per-cell original residual status is
voted before the next solver stage. A reported solution undergoes an additional
full original residual evaluation and finite L2 stopping test before outcome
consumption and output copying. Inspection receives these paths; it does not
replace multi-rank fault injection or resource-failure execution.

The author's public runtime tests observe `history_global` results of both
solves and compare independently evaluated original residuals. Their finite
initial data, domain-error callback and exhausted-iteration cases distinguish
solve failures from bind refusal, and snapshot accepted fields/history/clock.
Those runtime assertions were inspected and emitted, but not executed here.

## Commands and exact limits

Explicit production source import was verified as this checkout's
`python/pops/__init__.py`. The final source/math command uses the existing
interpreter without installing anything:

```
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python OPENBLAS_NUM_THREADS=1 \
 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q \
 tests/review/test_sol61_spatial_field_cda_independent.py \
 tests/review/test_sol61_spatial_nonlinear_math_oracle.py \
 tests/python/unit/fields/test_nonlinear_mixed_field_problem.py \
 tests/python/unit/fields/test_m27_mixed_public_source.py \
 tests/python/unit/fields/test_m27_projection_independent_review.py \
 tests/python/unit/time/test_implicit_stage_request.py \
 tests/python/unit/time/test_implicit_diffusion_request.py \
 tests/python/unit/codegen/test_coupled_implicit_codegen.py \
 tests/python/architecture/test_import_graph.py
```

Result: **97 PASS**, 61.69 seconds, exact `008bd383` sources. Ruff and diff checks
pass on the independent additions. Four real legacy cases, each resolved/emitted
in separate Python processes against an authenticated `git archive 2599d221`
and the candidate, retain identical canonical IR identities and complete emitted
C++ SHA256 pairs: mixed linear,
permuted mixed linear, implicit stage, and nonlinear-map implicit stage.
The helper is `tests/review/sol61_spatial_field_legacy_parity.py`; its local
receipt is `outputs/sol61-t3-independent-cda/legacy-parity.json`.

The independently emitted complete C++ body is retained under
`outputs/sol61-t3-independent-cda/complete-five-unknowns.cpp` and is syntax-checked
with the real source headers, `POPS_NATIVE_DIM=2`, Kokkos/MPI/OpenMP and shared
exception ABI defines. The `008bd383` body is byte-identical to the already
syntax-received `cda144e9` body. No generated large body or old output is committed
in this review.
The complete 61,959-byte body SHA256 is
`87f6b74eaeb8fcbd8fce9c6fa0334168f835dce77e356baf861200f26a7a9e28`.

The root's subsequent conditional Program IR8 promotion is a separate source
change and is not attributed to these SHAs. This review receives no installed
package tail, native Newton/GMRES convergence, real MultiFab solve, MPI execution,
GPU, checkpoint codec, complete nonlinear M27, T5 capture extension, or PDE
end-to-end qualification. Those remain central exact-source/artifact receipts.
