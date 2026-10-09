# Consumed provider instance contract

`consumed-provider-instance@1` separates a Module declaration from its native
storage address when consumed Field observations publish into several Case blocks
that instantiate the same Model. The declaration still owns the original exact
ProviderPack, operator, component contract, typed dependencies and scalar slots.

The resolved operation evidence carries exactly these fields:

```json
{"contract":"consumed-provider-instance@1",
 "definition_owner_qid":"<canonical Module owner>",
 "instance_owner_qid":"<canonical Case/block/model-definition owner>"}
```

The instance must authenticate as that definition's Case block. Every resolved
operation must belong to the same instance. Resolve qualifies the complete group
of repeated definition owners when any member has a consumed publication. This
includes siblings without a publication; it does not borrow another sibling's
published storage. A missing required runtime input remains a bind refusal.

Only local native `ComponentKey.owner_qid` changes. The other three strings remain
exact. Foreign Case-owned keys remain unchanged. Publication output keys and
provider identities, dependent provider keys, native Model and Program consumer
plans, InputAux bind keys and retained continuation keys all use the same
projection. Provider declarations are installed per qualified instance. Formula
carriers retain an immutable, witnessed contract; emitting under another block's
owner or revoking/mutating the projection refuses.

The existing native registry allocates distinct component addresses for the four
string keys. Compatible shapes can share a storage group while using disjoint
scalar slots. Its existing carrier allocation, Kokkos views, publication
transactions, finite checks and accepted/candidate rollback remain authoritative.
There is no new per-model buffer copy scheme or C ABI. ABI version 10 is unchanged.
Resolved evidence and carrier witnesses incorporate this local contract so a
compiler cache cannot silently substitute another instance.

Unique-owner publications and repeated definitions without consumed publications
retain the legacy behavior. A replay loads the modified modules' actual base Git
bytes and compares the complete generated Program and two Model sources for the
existing unique-owner witness: all three files remain byte-identical.

## Reproduction and scope

Run from the dedicated checkout with the existing read-only local environment:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python \
 tests/review/sol61_provider_instance_validate.py /absolute/new/evidence-directory
```

This compiles ten actual generated Program/Model translation units and executes a
small real-header Host test using ExactAuxiliaryRegistry, MultiFab, Kokkos kernels,
host mirrors and the existing transaction API. Three instances share component
names, publish values 1/3/5, derive their own values 3/11/27, reject a NaN candidate
without changing accepted values/generation, then accept a retry. Non-square
7×3 and 2×11 layouts and reversed registration order are covered.

Source tests use three instances with permuted declarations and operator inputs,
two distinct stage fractions and an evolving declared forcing carrier. A separate
joint elliptic witness publishes different unknowns to siblings and makes the
beta instance the actual first provider consumer. The public Reaction equations
have one shared constant kernel, preserving SharedMeanGauge's actual contract.
An independent NumPy reference checks discrete equations and distinguishes the
wrong sibling's field. Two-cell x sampling makes the selected cosine mode zero;
the nontrivial oracle control uses 3×11, while 2×11 remains a structural/Host layout.

The original failed authoring script and traceback are retained externally as
`field-publication-negative-7586.py` (SHA256
`d2cbd64af6ce7a3e4583b670162384fdc384e7550df8d048be835dfdb513339b`).
The derived fixture authors its forcing source before attaching Aux providers to
the final Module, fixing a separate fixture cache invalidation without changing
the forcing equation. The baseline hook still reproduces the original cardinality
refusal on this corrected fixture.

The prospective installed tests are
`tests/python/integration/runtime/test_field_publication_instances_runtime.py`.
They authenticate installed Python/native origin and the published typed ABI,
retain complete C25 for every Model and Program before bind, and save initial plus
two accepted states through NPY/CP9 with clocks/cursors. They compare the unchanged
stage equations to an independent discrete reference, preserve catalyst bytes and
use an absolute `2e-12` guard. They require a fresh ROOT build/install and have not
run. Source/Host results establish no PoPS Native, MPI, AMR, GPU, restart or full
BGK scientific qualification.

The broader Uniform restart fixture currently omits `pops_checkpoint_version`:
19 failures reproduce on the unchanged base and on this candidate. Both original
receipts remain preserved; this change does not alter that reader or fixture.
