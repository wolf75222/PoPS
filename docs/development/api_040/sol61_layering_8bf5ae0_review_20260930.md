# Independent reception of the second FiniteLinear identity fix

Date: 30 September 2026. Author fix: `8bf5ae0c3e028aefcf0b57f70a195017a896edfd`.
Receiver: `ff07a7ce`, consisting of 908508a + independent reviews + 1c84af0 +
8bf5ae0, without the author's unrelated diagnostic or native changes.
Exact pre-layering comparison parent: `bc37b0af4012a10fa3e9bd7ecaccd1f7befc897b`.

The independently reproduced residual from the first reception is now closed:
`(vec + vec * Fraction(2, 3)).materialize(...)` has the exact parent encoding,
application sharing, and actual `Program._ir_hash()`,
`c7f6fdecbb652608c4fe67afeb38ed9bda28c72a7d4fd10ee2dba1ac7205aed1`.
Plain and mixed programs also retain their parent identities. All finite fixture
encodings match, rather than only their expanded mathematical expressions.
Literal multiplication captures its payload once but retains distinct per-component
declaration identities; symbolic operands keep their existing sharing.

The frozen probe `tests/review/sol61_layering_8bf5ae0.py` authenticates the relevant
production files against the author commit and executes immutable archives of both
revisions. It reuses the independent probes from review commits e0062b1 and 7c03184
with asserted substitutions. Result: 33 comparative checks pass, including ten
explicitly refused conditions. No forbidden reverse import was introduced; the
unchanged architecture gate passes. Historical differences in truth-value exception
subclass/message and infinite-input wording from 908 remain recorded: this review
does not normalize them away or claim identical exception wording.

The selected source suite, including nine independent weak-cache lifetime/refusal
checks and thirteen vector-reduction witnesses, passes **152 tests in 13.99 s**,
without skips. Weak declaration and IR lifetimes, repeated projections, shared Const,
malformed protocols, cycles, immutability and explicit refusals remain exercised.

Reproduction in the exclusive review checkout:

```sh
rtk proxy /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_layering_8bf5ae0.py
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/review/test_sol61_finite_weak_cache.py tests/review/test_sol61_vector_reduction_oracles.py tests/python/unit/numerics/test_symbolic_policy_layering.py tests/python/unit/numerics/test_finite_linear_maps.py tests/python/unit/time/test_program_expressions.py tests/python/unit/time/test_program_expressions_adversarial.py tests/python/unit/numerics/test_user_reconstruction_adversarial.py tests/python/unit/numerics/test_user_joint_reconstruction_independent.py tests/python/unit/numerics/test_user_face_authoring.py tests/python/unit/numerics/test_user_face_independent_review.py tests/python/unit/codegen/test_exact_scalar_literals.py tests/python/architecture/test_import_graph.py tests/python/architecture/test_time_package_layout.py
```

The raw receipt remains in `outputs/sol61-layering-8bf5ae0/receipt.json`.
Parent Python archive SHA256:
`b2a8d77345da0cf03f0f297c3543ac8f7838c60686e101a7d5e65b0f10a53d53`.
Receiver Python archive SHA256:
`2611c01c93613572adcd6737de345254e24c3a897ca91ad453eab6bbc5d4daf9`.

Source reception is positive for this bounded identity fix. No package installation,
native trajectory, MPI/GPU, PDE qualification, header or SDK mutation is claimed.
