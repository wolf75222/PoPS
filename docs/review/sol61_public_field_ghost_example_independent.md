# Independent Source review of public Field/Ghost example

Author freeze: 016dc020b70417f9b63a4382745ec46ee139d15f. Reviewer: GPT-6.1 Sol. Verdict: bounded Source preparation accepted; no Native execution or scientific reception.

The module defines the physical equations before numerics and execution: dc/dt=0, dm/dt=m, zero transport flux, and -laplacian(phi)+8phi=8m with homogeneous Neumann boundaries. The xmin expression uses the explicit primary-state InteriorTrace and the independently declared Field. Inferred component metadata delegates external values through the authenticated component route. Installed package paths, ABI8 and Serial Dim2 are checked before execution; three authentic checkpoint calls and a fresh bind/restart are present.

The example verifies valid-state bytes, fine topology and accepted time after restart. It does not compute the independent composite Original F <=1e-10 oracle added in fixture 8179f04e, compare full-grown Ghost storage, authenticate nonmutating Field freshness, or compare restarted Field payload bits. The receipt and README explicitly disclose these limits. Therefore the example's restart checks cannot substitute for that fixture's scientific guards. The six README contracts describe source routing and interfaces, without promoting preparation to a Native result.

Independent structural counterexample: rename the domain/model/case/block/Field public identities, preserving equations and supports. Both actual Source resolve and detached boundary compilation retain two expressions, exactly one inferred Field dependency, no extra State dependency, and one external region; the authenticated manifest identity changes. A valid identifier is required for the renamed Field output by the existing generic provider authoring contract.

Validation (execution tail never evaluated):

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider -o pythonpath=python tests/review/test_sol61_public_initial_field_ghost_example.py tests/review/test_sol61_public_initial_field_ghost_example_independent.py -q
```

Result: 3 passed in 9.49s, zero skips. No JIT, Native SDK execution, ENV mutation, ROOT seal, or scientific approval.
