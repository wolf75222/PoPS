# Execution admission: Native wire widths

Hooke / GPT-6.1 Sol; supplement to f1b32ede. Independent review found that an
open execution protocol could supply a numerator/denominator beyond int64, pass
Python admission, install boundary/provider callbacks, then fail conversion at
the real `set_temporal_relations(vector<int64_t>, vector<int64_t>, ...)` binding.

The shared contract now rejects numerator or denominator above INT64_MAX before
callbacks. It requires the already-reduced ratio emitted by AMRClockRelation's
existing Fraction authoring contract, avoiding silent Native normalization of
an open protocol's different identity. Positive integer/type/ratio/remainder
checks remain intact; no physical-model range is added.

Runtime installation additionally requires the artifact's real resolved dimension
to be an exact integer1/2/3. None cannot bypass ranked extent/int32 admission.
There is no guessed dimension or fallback. `AmrSystemConfig.level_count` is int:
the exact hierarchy transition count is checked against that range before
synchronous relation allocation. Parent/child indices continue to be generated
from that hierarchy and validated as exact adjacent integers, not foreign labels.

Source tests cover overflowing numerator/denominator, non-reduced2/2, missing
dimension, and an adversarial oversized transition count with zero callbacks.
The valid INT64_MAX numerator uses a real public resolved plan and InstallPlan;
its clock sink is an explicit Source spy and allocates no Native grids/substeps.
This does not claim that executing that many substeps is practical or qualified.

Affected cohort after int64/dimension changes: **208 PASS in57.92s**, same eight
test files/command as f1b's note. After the additional level_count range guard,
the final focused execution-contract file passes separately (result below).
No Root/ENV/cache/JIT/Binary write, C++ build, Native campaign or SCI seal.

Final focused command: `rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops/bin/python -m pytest --noconftest -o pythonpath=python tests/python/unit/runtime/test_amr_execution_contract.py -q --tb=short` - **19 PASS in21.37s**.
