# Generic multi-species

No species is hardcoded. A species, fluid, population or evolved field is a named
**BlockInstance** of a {doc}`StateSpace <typed-ir>`; the core knows BlockInstances,
StateSpaces, operators, a Program and bindings, not "electrons". The same StateSpace can be
instantiated several times (`fluid_A`, `fluid_B`, ...).

## Spaces and instances

```python
import adc.model as model
e = model.StateSpace("electron_state", ["ne", "mex", "mey"],
                     roles={"ne": "Density", "mex": "MomentumX", "mey": "MomentumY"})
i = model.StateSpace("ion_state", ["ni", "mix", "miy"])
n = model.StateSpace("neutral_state", ["nn", "mnx", "mny"])
```

## Arbitrary-arity operators and typed multi-output

An operator takes an arbitrary number of states (1, 2, 3, N) and may return several typed
outputs. `adc.model.RateBundle` is the typed multi-output of a coupling: one
`Rate(StateSpace)` per participating block, of arbitrary arity. It is typed, so a wrong rate
on a wrong state is rejected:

```python
coll = model.RateBundle({"electrons": model.Rate(e), "ions": model.Rate(i),
                         "neutrals": model.Rate(n)})   # arity 3
coll["electrons"]                # RateSpace('Rate(electron_state)')
coll.require("electrons", e)     # ok
coll.require("electrons", i)     # TypeError: wrong rate on wrong state
```

A coupling (charge density, a field solve, collisions, ionization, radiation) is just an
ordinary multi-input / multi-output operator; the runtime has no special "coupling" category.

## Multi-block program and atomic commit

A Program references several blocks, solves coupled fields from their stage states, and
commits them atomically (no operator observes a partially committed coupled group):

```python
import adc.time as adctime
P = adctime.Program("three_fluids_step")
dt = P.dt
e_n, i_n, n_n = P.state("electrons"), P.state("ions"), P.state("neutrals")
fields = P.solve_fields_from_blocks([e_n, i_n, n_n], name="fields")   # coupled, arity 3
e1 = P.linear_combine("e1", e_n + dt * P.rhs(name="Re", state=e_n, fields=fields, flux=True))
# ... i1, n1 ...
P.commit_many({"electrons": e1, "ions": i1, "neutrals": n1})          # atomic
```

`adc.time.StageStateSet` (built by `P.state_set`) packages a coherent set of stage states so
a field solve reads an unambiguous version of each block (see
`examples/spec3/stage_state_set_field_solve.py`).

## Status

The operator-first multi-block kernel is in place at the IR level: multiple StateSpaces on a
`Module`, a multi-block `Program` (`P.state` per block, `solve_fields_from_blocks`),
`RateBundle`, `commit_many` (atomic, validated). `examples/spec3/multispecies_three_fluids.py`
builds a 3-species step. The board sugar (`m.species` for N > 1, `m.coupled_rate` -> a
`RateBundle` operator) and the RUNTIME for a coupled multi-block field solve / a coupled-rate
operator (codegen + execution) are tracked by ADC-457; a single-species board model is
authored today via {doc}`board-like-dsl`.
