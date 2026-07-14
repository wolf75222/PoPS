# Final advection-relaxation IMEX + AMR contract

[`EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py`](../../examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py)
is the executable acceptance target for the complete
operator-first path. It has one runtime route:

```text
Model -> Case -> validate -> resolve(AMR) -> compile -> bind -> pops.run
```

The example does not select a named native time stepper or a separate AMR executor. Its executable
acceptance path compares the manual `Program` and `pops.lib.time.IMEX` spelling node-for-node, checks
their normalized semantic data and identity, then requires the same accepted runtime snapshot.

## Exact additive method

The method is the rational CN/Heun pair

```text
A_exp = [[0, 0], [1, 0]]       b_exp = [1/2, 1/2]    c_exp = [0, 1]
A_imp = [[0, 0], [1/2, 1/2]]   b_imp = [1/2, 1/2]    c_imp = [0, 1]
```

Every coefficient is authored as `Fraction`; the stage points retain distinct `explicit` and
`implicit` coordinates even where their values happen to agree. The graph contains the predictor,
operator calls, local diagonal solve, partition rates, final combination and one commit. Order is a
property of this graph and tableau, not a repeated `order=2` option.

Every fallible public solve returns an unreadable `SolveOutcome`. The example consumes every field solve with
`RejectAttempt()`. A failed solve therefore raises the typed native rejection signal before a field,
state, diagnostic or output can read a partial result. Local affine elimination remains a value
operation because it has no iterative outcome to classify.

`Model.field_operator(...)` declares the physical equation and its RHS providers. The sole callable
time-Program authority is the `FieldHandle` returned by `Case.field(operator, discretization)`: both
the manual graph and `pops.lib.time.IMEX(fields_operator=...)` receive that exact handle. This keeps
the field outputs, numerical plan and native install route under one Case-owned identity.

## Transaction and hierarchy

The AMR Program driver owns the accepted-state boundary. A hierarchy attempt stages and restores:

- level state and clocks;
- coarse/fine flux ledgers and reflux contributions;
- history rings and their flux publications;
- regrid-dependent synchronization state;
- field materializations and consumer schedule cursors.

The driver commits only after subcycling, synchronization, reflux and average-down complete. A
`RejectAttempt` unwinds the same hierarchy transaction. The example deliberately does not pass
string guards or projections to `Program.step_strategy`: those strings are report metadata, not
executed numerical contracts. Executable guard/projection composition belongs to the typed ADC-666
contract and must not be simulated here.

## AMR authorities

The adaptive layout owns:

- exactly two levels with one explicit two-to-one transition (cumulative refinements 1, 2) and
  subcycled execution; this is the installed provider's executable composite-field envelope;
- strict above/below refinement and coarsening predicates;
- a discrete gradient predicate resolved against the selected FV stencil;
- explicit hysteresis/equality/conflict semantics;
- conservative state prolongation, restriction, coarse/fine fill and time interpolation;
- elliptic recomputation after regrid instead of interpolating a stale solved field.

Reflux order, halo depth and proper nesting are consequences of the resolved FV and transfer
providers. They are not repeated as user integers. The diagnostic elliptic solve carries a distinct
`FieldContext` for every IMEX stage, so a field solved from one stage cannot be read as another.

The static transport-boundary classification is consistent with the declared positive velocity
domains: minimum faces are inflow and maximum faces are outflow. Supporting signed runtime
velocities requires a characteristic boundary provider that switches the incoming subspace; this
example does not claim that capability through a fixed boundary table.

## Accepted-only effects and strict restart

HDF5, NPZ and ParaView are independent `ScientificOutput` consumers on `AcceptedStep` schedules.
The checkpoint consumer uses the accepted-state restart provider with `bit_identical=True`. Rejected
attempts neither publish files nor advance consumer cursors. Restart authenticates the artifact,
bind inputs, Program graph, layout/hierarchy, transfer providers, parameter values and accepted
temporal/consumer state before restoring data; there is no weaker fallback policy.

Run the acceptance target with

```console
python examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py --output-dir /tmp/pops-imex-amr
```

The command reopens the emitted HDF5 and ParaView files, retains a real accepted-state checkpoint and
restarts a fresh bound simulation from it. It compares time, macro-step, every AMR level of every
qualified conservative state and solved-field route, patch topology, Program/consumer identities and
consumer cursors bit-for-bit. It then advances the uninterrupted and restarted instances once more
and repeats the complete comparison before exercising the preset parity run. A printed success
therefore follows real I/O, restart, continuation and manual/factory checks; it is not a demonstration
placeholder.
