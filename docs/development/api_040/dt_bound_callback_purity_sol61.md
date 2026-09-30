# Dt-bound callback purity, including non-SSA mutations

Follow-up to `9c4209a5e5ecb05ac71a62f4d2c21b6e3acd4300`, integrated as
`9e0ee22f`. The independent review `893379b4255711135b1d1a8ab2e0e896bbc542a1`
found a pre-existing hole: `P.commit(passive.next, candidate)` can add a physical
publication without adding a node to the bound DAG. Read-only operation traversal
alone therefore cannot establish callback purity.

The existing identity-preserving `_AuthoringSnapshot` now authenticates the
callback's net authoring effect. The query may allocate issued nodes, region IDs,
and new state/field binding entries; previously present mapping entries and nested
regions must remain unchanged. SSA and region counters may advance. The first
state read may establish its exact live Case authority. Every other Program
attribute and original mutable container is protected by default, including
future attributes not covered by this allocation whitelist. No equality overload
on ProgramValue is used to authenticate original records.

`set_dt_bound` executes its callback inside that query guard before publishing
the bound. A mutation refuses with a `ValueError` naming the changed metadata.
The existing authoring snapshot restores original containers **in place**, plus
their attribute bindings. Its enclosing atomic transaction still covers later
region/DAG validation. Pure reads, query-only blocks and shared captured DAGs
retain the prior behavior and fingerprints.

The public hidden-commit counterexample now refuses and preserves the sole
original commit, original serialized graph and SSA allocation counter. A valid
retry can read the passive state without granting it a publication. Additional
public tests refuse strategy, cadence, cell-local-time, integral declarations,
stage identity, history configuration and retiming of pre-existing graph values;
they verify exact restored graph/counters and a successful pure retry. Freeze
inside the callback remains refused by its pre-existing active-region guard.

Source reception: **49 passed in 12.66 s**. This comprises the coherent authoring,
codegen, provenance and readonly-input suite, plus the unchanged independent
source review from Ptolemy's checkout. That review also uses an actual git archive
of the old source to compare complete legacy IR, semantic data, hash and C++
bytes. The six fixed author fingerprints remain identical. Ruff and staged
whitespace checks pass. No native test, JIT, C++ build, header or environment
mutation occurred.

The source run uses `env -u PYTHONPATH`, the explicit `pops-api040` Python with
`-I`, and prepends only this checkout's `python` and repository roots. Pytest runs
the four files listed in the preceding capture-closure report plus
`PoPS-sol61-dt-bound-review/tests/review/test_sol61_dt_bound_capture_independent.py`,
with `-p no:cacheprovider`. The independent test's public fixture helper was
copied byte-for-byte into this checkout as an **untracked source-only import**;
the fixture's native tests were not executed and are not included in this commit.

This guard authenticates Program authoring state, not arbitrary effects a Python
callback could perform outside the Program. Native execution of the bound and
collective runtime refusal remain for root's reconstructed package and Ptolemy's
public native reception. The legacy native scalar-admission contract is unchanged.
