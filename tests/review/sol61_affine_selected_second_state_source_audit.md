# M16 selected second State — Source counter-case

Author checkout `b9e6daf3386291bf5df19c30bfa6f0bb2b9bb974` (M16 e943 retained).
No production edit, Native execution, JIT, build, ENV mutation or runtime receipt.

Result: **the authentic resolved Program path emits the explicit affine map for
an independently selected second State**. The counter-case declares two public
`Model.species` StateSpaces, each with six differently named components. The first
is committed unchanged. The second uses a permuted degree2 basis and a bijection
whose slot order differs from both basis order and declaration order. Its operator
is declared through public `Module.operator`, with the actual second StateSpace
in `Signature((), LocalLinearOperator(space, space))` and a typed Const matrix.
Case blocks explicitly select their States with `states=(state,)`.

The test performs real validate → resolve → selected lowering → detached full
ProgramGraph C++ emission. It proves the affine source's FieldView reads
`ctx.state(the authenticated second-block index)`, all gathers use the second block's six local slots, and the exact
selected emitter component names match that State. No fabricated emitter carrier
or overridden registry is involved. Full graph identity is retained by detachment.

The suspected count check therefore has legitimate scope in the whole Program
pipeline. `ProgramModelGraph.from_resolved_blocks` (program_models.py:151) lowers
with each block's authenticated `state_spaces[0]` and operation plan; its
`model_for_node` (line264) routes by exact block. The affine emitter receives that
selected implementation, not a twelve-component concatenation. The common
`_cell_locals` FieldView binding uses those selected `cons_names` as local component
indices. A separate native block is the layout authority, so no new StateSubset
contract is required for this tested route.

A **distinct genuine facade defect** is reproduced before affine authoring:
`Model.operator(... returns=Model.local_linear_operator(..., on=second))` raises
`KeyError: unknown operator 'selected_skew'`. In board.py:1078 the matrix arity
still comes from the old first-State DSL; in board.py:1118 the registration goes
through that DSL, but `_registered_operator_handle` reads the multi-module registry
(board.py:1187). With equal arities this reaches the wrong-registry refusal; unequal
arities would encounter the earlier matrix-size gate. This is not evidence that
Native affine execution selects the first species. The source-only test preserves
that exact refusal without patching or bypassing the validator.

If ROOT assigns a facade repair, the bounded direction is to derive matrix domain
and range from the explicit `on` State, register the existing typed LocalLinearOperator
in the actual multi-module authority, and reuse selected lowering. Existing operator
signature, identity and provider ownership machinery suffices; a special first-species
branch or new numerical recipe would be inappropriate. Whether release metadata
requires a version change belongs to ROOT. No repair is made in this audit.

Both States use the same zero characteristic law to respect the existing exact
multi-species eigenvalue-law contract. This does not qualify differing eigenvalue
laws, dynamic Field providers, Native storage, multi-level AMR or MPI.

Reproduction:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest -q -o pythonpath='python .' tests/review/test_sol61_affine_selected_second_state.py tests/python/unit/codegen/test_multistate_selected_emitter.py
```

Cohort result: 8 Source PASS43.91s; the strengthened exact second-block FieldView
assertion is checked against the Program block-index authority, not the model
declaration position. Blocks are numbered by first Program use (second State is
index0 in this case); this does not select the first physical State. The focused two-test rerun passes in3.92s. Source package origin and `_pops`
absence are asserted. No Native/backend qualification follows from this result.
