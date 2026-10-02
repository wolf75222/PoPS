# Typed multi-State local linear facade repair — Source @1

Parent `9534edbe13ad2e83126874716a9be75fb91db99b`; previous M16 gels preserved.
No Native/JIT/build/ENV/Main changes or scientific qualification.

The public counterexample uses `Model.species` of sizes2 and6. Before repair,
`model.local_linear_operator(...,on=second,matrix=6x6)` refuses with
`ValueError: local_linear_operator('selected_skew') needs a 2x2 matrix`.
The exact reproduction was run before the production edit (1FAIL0.94s; retained
private `/tmp/sol61-multistate-operator-red.log`). Equal arities formerly reached
the distinct wrong-registry KeyError captured by the preceding Source audit.

The mathematical object now derives multi-State matrix arity from its exact
explicit `on` handle. Missing/foreign States and wrong shapes are rejected before
publication. Mono-State retains the original optional-on path and legacy DSL
registration. For multi-State registration, the body is converted through the
existing expression protocol and published in the actual Module registry with
`Signature(Fields?, LocalLinearOperator(on.space,on.space))`. Field inputs must
be distinct actually declared FieldSpaces; the existing signature validator
continues to require zero or one FieldSpace and a square operator. Registry and
facade metadata publication uses the existing atomic authoring mechanism. No
physical name, species order, canonical slot or matrix-size recipe selects the
scope. Public Module registration remains available unchanged.

The genuine full Program test revealed one accompanying typed scope omission:
selected lowering looked only at `signature.inputs` to filter State-specific
operators. LocalLinearOperator's State lies in its typed output.domain/range,
while its inputs are Fields only. The existing filter now also uses that domain;
it avoids installing the six-by-six second-State matrix in the two-component
first-State emitter. This is a three-line common typed lowering change, not a
model branch. The signature registry already requires domain==range. Per-block
FieldView and Program block-index authorities remain unchanged.

The new Source tests run real validate → resolve → detached full ProgramGraph
C++ emission for the selected second State. Both emitters are checked against
their distinct component arities. Other tests cover first-State explicit scope,
real typed Field dependencies, missing/foreign/wrong-size/unknown-input refusal
with unchanged Module identity, and mono optional-on/explicit-on Module identity.
The earlier dated audit's facade-negative test is updated to a positive typed
registration test; its original refusal remains in immutable Git history.

No new wire, ABI or IR shape is introduced: the existing LocalLinearOperator
signature/body is now correctly authored by the facade, exactly as a public
Module could already declare it. Affine Program still uses conditional IR23.
Source/package identities change and require fresh installed authentication;
this report does not inherit an old Native/SDK receipt.

Scope: declarations made in the actual current model mode. This change does not
extend migration of previously registered DSL operators when a later species
promotes a single-State model, nor does it add support for conflicting global
wave laws. These are separate authoring/promotion contracts. Field-dependent
registration is tested; no real dynamic Field solve or Native matrix application
is claimed here.

Reproduction:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest -q -o pythonpath='python .' tests/python/unit/physics/test_multistate_local_linear_operator.py tests/python/unit/physics/test_physics_board.py tests/python/unit/time/test_operator_handle_resolution.py tests/review/test_sol61_affine_selected_second_state.py tests/python/unit/codegen/test_multistate_selected_emitter.py
```

Final affected cohort: **40 Source PASS45.02s**, zero skips. New full pipeline
asserts genuine Source package origin and `_pops` absence. Independent review and
ROOT installed Native execution remain separate obligations.
