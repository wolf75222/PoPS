# Independent review: read-only State bind inputs

Reviewed the root's `inspect_compiled.py` read-input discovery change after
`d16b2e4` and `a85a70b`. File SHA-256 at reception:
`3b57fc61f582c736efc6ecfd17bc6dd55eab5c4465c73d4927d98984c90592cd`.
No production file was edited by the reviewer.

The union of committed state handles with the exact handles of `state` operations
is the right distinction: an initial value is required even if the Program only
reads it. The traversal covers the existing nested condition/body/branch/apply/
residual regions, plus the separate timestep-bound subprogram. It does not change
the commit set, create a target unknown, execute an inactive branch or weaken
state qualification. A missing state identity or absent exact block metadata is
still refused. Per-layout inspection uses its existing exact metadata partition.

The independent test
`tests/python/unit/codegen/test_readonly_control_input_review.py` builds and resolves
a real public Case with four states. One is published, one is read exclusively
inside two nested lazy branches, one only inside `set_dt_bound`, and one is unused.
It checks that the two read-only states really are absent from the top-level
state operations, that their exact serialized identities become required bind
inputs, that the unused block is excluded, and that the Program hash and sole
publication are unchanged by inspection.

The discriminator was replayed with the exact `_build_arguments` function from
`d16b2e4` and the current metadata helpers: it fails because only `dual` is reported
(1 failed, 1.03 s). A first attempt against the entire older isolated package
stopped earlier on its unrelated missing ghost-depth metadata; that attempt is
not used as evidence of this input-discovery defect. The test passes against the
reviewed MAIN sources (**1 passed, 2.16 s**) with
`env -u PYTHONPATH` and explicit pytest source path. No native module, JIT, build
or installation was needed. This is source-level authority evidence; execution
and read-only target immutability still require the central installed reception.

Verdict: favorable within this bounded discovery change. No additional production
guard was required by the control-flow counterexample. The test commit depends on
the root's read-input discovery fix and is intentionally red on the review branch's
older `inspect_compiled.py`.
