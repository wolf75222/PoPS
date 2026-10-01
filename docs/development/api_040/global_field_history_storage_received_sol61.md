# Independent reception of the global history storage port

This reception starts from production `0bc86f3e91d54137decbf9524ab07ffac8a90012`
and fixture/documentation `4c7412a83d54dce14dac5f6abfdb8c17de91128a`, read in an
exclusive worktree. No installed package, native build, JIT, environment or
principal worktree was modified. The date is 1 October 2026.

The independent builder defines three global unknowns and their original
reaction/diffusion equations, with captures from three separate physical
blocks. A fourth real Case block supplies an already declared TimeState solely
for storage. Widths three and five discriminate storage layout from the scalar
observation's physical width. No author fixture or manufactured-value helper is
imported by these probes. Newton and CG are declaration descriptors here;
neither solver is executed by the Python probes.

## Open findings on the first production gel

1. **P2 — the public linear route cannot decode its frozen field identity.**
   `storage_contract` reconstructs the handle from a `field_problem_load` node's
   immutable attributes. `OwnerPath.from_data` receives a tuple of nodes and
   rejects it because its wire protocol requires a list. Both TimePoint and
   StagePoint linear witnesses fail before the storage contract is issued.
2. **P2 — a real StagePoint is treated as a TimePoint.** A valid public
   `Program.stage(..., c=0)` has partition coordinates and `.time`, but no
   `.clock`. The nonlinear witness reaches the direct `.clock` access and raises
   AttributeError. Its exact coordinates must be resolved through the declared
   stage contract; they must not be relabelled onto an arbitrary clock.
3. **P2 — the original storage owner is not independently authenticated.**
   Replacing the store's block with another issued same-Case/frame/clock block,
   recomputing `global_field_storage`, and replacing `_history_blocks[name]`
   coherently is accepted by `_ir_hash`. The expected contract is recomputed
   from those mutable projections. Removing only `global_field_storage` is also
   accepted and loses the IR16 qualification. These are pre-emission authority
   failures, not assertions that a terminal physical solve accepted a bad state.

The regression tests require refusal of the latter two attacks. On the first
gel they are red; no xfail, weakened guard, or new legacy interpretation is used.
The ordinary public attempt to reuse a ring name with another valid owner does
already refuse. Individual descriptor mutations also refuse; those facts do
not close a coherent reseal or disappearance of the discriminant.

## Separately received native guard branching

The C++ host probe extracts the actual `store_global_field_history`, exact
point/layout/owner methods, `require_prepared_lane_`, and
`prepare_history_mutation_collectively_` from the frozen headers. It compiles
them with clang++ C++20 and two host threads. Field storage, reduction and
transport are explicit substitutes; `store_history_` is a publication counter.
This is not a Kokkos/MPI or numeric rollback reception.

Forty-one scenarios pass: a valid width-five prototype/scalar ring, thirty-nine
negative or disagreement cases injected on host rank zero only, and a legal
empty peer. The negatives cover unknown/foreign owner, immutable prepared State
witness, owner/descriptor tables, ring depth/components/layout/distribution/local
rank/ghosts, nonfinite valid cells, exact clock/tick/level/substep/fraction/dt/time,
foreign evaluation metadata, launch/fence/contract-allocation failure, and exact
operation disagreement. Both peers refuse before the counter publication. Local
preparation failure is voted before operation agreement. The coherent native
table reseal is refused with the original emitted State witness unchanged.

This native branching result does not authenticate a newly forged CPP witness:
the Python pre-emission proof must bind the witness initially issued for that
ring. Nor does it receive the actual `store_history_` transaction, regrid,
checkpoint/restart, device kernels, MPI collectives or empty MPI ranks.

## Commands and initial results

All commands are prefixed by RTK. The interpreter is the existing
`pops-api040/bin/python`, selected with `PYTHONPATH=python` and
`PYTHONDONTWRITEBYTECODE=1`; the installed implementation is not tested here.

```
rtk proxy env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest tests/review/test_sol61_history_storage_owner_received.py -k 'not public_linear_and_nonlinear and not resealed_existing_owner' -q -p no:cacheprovider
```

Before addition of the downgrade countertest, that selection was **20 passed,
5 deselected**, including the C++ host probe. The full first ten-case public
selection was **6 passed, 4 failed**; the separately added downgrade probe was
red with `DID NOT RAISE ValueError`. The earliest harness construction errors
(`StateHandle.frame`, unresolved `FieldProblem.to_data`) were corrected in the
independent helper and are not production findings.

The coherent final first-gel selection contains **26 tests: 21 passed, 5 failed
in 37.46 s**. Its raw log is retained privately at
`outputs/sol61-history-storage-independent/first-gel.log`. The five failures are
the two linear profiles, the nonlinear StagePoint profile, coherent owner
reseal, and removal of the contract discriminant. The scalar nonlinear
TimePoint profile succeeds.

## Fresh comparison with the option omitted

The separate parent worktree is exactly
`a0f92caad8da3c5c1e6bd99b58aff7cf07573e5b`. Two fresh interpreters load the **same
absolute helper path and callsites**, one against that parent's Python source
and one against the first candidate. The helper's `__main__` emits the complete
resolved Program payload (including provenance), full C++ string, IR hash and
three Module hashes for linear CG and original-field Newton. Both have a legacy
unowned global history and real public Uniform resolve/emission, without the new
option. There is no provenance normalization, helper relocation or omitted CPP.

The complete JSON files are byte-identical by `cmp`, both SHA256
`d7f10e2d73fd2aa8825cc3a4d78290c3c94095fa604ae56407d7069de0b0cd1d`.
They are retained as `parent-default.json` and `candidate-default.json` beside
the first-gel log. This receives two independently authored legacy profiles;
it does not requalify the author's nine other profiles or a compiled artifact.

```
rtk proxy env PYTHONPATH=/Users/romaindespoulain/dev/tmp/PoPS-sol61-history-storage-parent/python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python /Users/romaindespoulain/dev/tmp/PoPS-sol61-history-storage-review/tests/review/sol61_history_storage_witness.py
rtk proxy env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python /Users/romaindespoulain/dev/tmp/PoPS-sol61-history-storage-review/tests/review/sol61_history_storage_witness.py
```

The initial parity helper's import spelling and direct State alias were corrected
before comparison: preserving a State uses a fresh `1*current` expression at its
next point, without retiming the original capture. Those setup failures were
not attributed to the new production code.

Native reception requires
the newly changed SDK header rebuilt and authenticated by ROOT. No previous
native history or Stage dataset qualifies this new port.
