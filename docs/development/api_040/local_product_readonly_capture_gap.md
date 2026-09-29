# One local unknown with a captured read-only State

Source audit for the M18 composition `4619a34`, against the local storage fix
`0ae3db0`. No native compilation or installed bind was performed.

The existing public `LocalResidual` product accepts a mapping with **one**
unknown State of width three and a captured State of width three from another
co-located block:

```python
def residual(P, z, *, target):
    return {"dual": tuple(z["dual"][i]**2-target[i] for i in range(3))}

outcome = P.solve(LocalResidual(residual, {"dual": seed},
    captures={"target": target_state.n}), solver=LocalNewton())
solution = outcome.consume(action=FailRun())
P.commit(dual_state.next, solution[dual_block])
```

The independent public fixture validates, resolves, builds its model graph and
emits the native provider with `prepare_local_nonlinear_problem<3>`. It has one
output and one commit. The captured State is an input, not an unknown. The
existing original-residual evaluator and collective co-location checks are used.
The three-variable polynomial is only a minimal capture probe; it does not claim
to qualify the entropy closure or its exponential domain.

The failure is downstream: `_build_arguments` in `inspect_compiled.py` enumerates
only `program.commits()`. It reports `instances={'dual'}` although the real
resolved Program also reads `target`. The test expecting both exact required bind
inputs is red. Consequently the bind validation gates reject supplied initial
data for the missing target as an unknown block. This cannot be repaired by
declaring extra algebraic unknowns: they change the system solved.

Source result: **1 passed, 1 failed**, with the failing assertion displaying
`{'dual'} != {'dual', 'target'}`. Reproduce with the configured source environment:

```sh
env -u PYTHONPATH python -m pytest -q -o pythonpath=python \
  tests/python/unit/codegen/test_local_product_readonly_capture.py
```

The shared correction belongs to argument inspection and its exact State
inventory, currently owned by the primary agent: include referenced read-only
State authorities as required inputs without adding them to the commit set.
Carry their identities, component widths and declared storage depth through bind;
do not include unrelated authoring blocks or derive identities from names alone.
Keep the stricter unknown-block and missing-depth refusals intact.

After that correction, M18 can retain only `{"dual": seed}`, capture `target.n`,
remove the target identity equations and publish only the dual result. The
original three entropy residuals remain authoritative. Native reception must
verify nonuniform targets remain unchanged, failed closure leaves both states
unchanged, and permutation/rebind retain the qualified capture. A scalar
LocalResidual's same-block restriction need not be weakened: the named product
already represents this operation.
