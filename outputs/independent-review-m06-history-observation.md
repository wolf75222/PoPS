# Independent review: M06 accepted history observation

Reviewed `28f9be5aedfbba082202519aa8a9e63c5d09131e` in the isolated `PoPS-numerical-bodies` checkout. This is a read-only source and receipt review; I did not rerun the native probe or the eight-scenario acceptance campaign.

## Verdict

The M06 correction is justified and narrowly scoped. The example diff changes only its two final diagnostic reads, `M06_temperature` and `M06_liquid_fraction`, from Uniform history slot 0 to slot 1, plus an explanatory comment. Its equation, constitutive conversion, parameters, guards, time step, initial data, and acceptance thresholds are untouched.

`HistoryManager::rotate(clock)` swaps the just-written working slot 0 into slot 1 on acceptance, including aligned sample and `dt` metadata. M06 registers each diagnostic with `depth=1`, yielding two physical slots. The Python `RuntimeInstance.history_global(name, level_or_slot)` forwards the second argument as the slot for Uniform layouts; the three-argument form belongs to AMR. Thus `history_global(name, 1)` is the latest accepted sample *after* M06's committed step, while slot 0 is recycled storage.

The committed 4×4 two-step probe checks every array entry against independent constants. Its receipt records the installed Dim=2 native SHA-256 `6b5f452e432976a94b69635b96aad1a04c42c8d5e933cad5cf785301089488fb` and ABI key. After the second sensible step, accepted `H=1.5625`, recycled `T0=0.765625`, and accepted `T1=0.78125`; after the second plateau step, accepted `H=2.5`, recycled `f0=0`, and accepted `f1=1/6`. These separate the slots and match the declared enthalpy law. The receipt and probe support a genuine installed run under the pinned artifact; I did not independently execute it or authenticate the wheel currently on disk. At the first step, cold-start fill makes both slots equal, so a one-step-only test would have missed the defect.

The full M06 eight-scenario example has **not** been rerun after this correction. This review supports the observation fix, not final scientific qualification.

## Adjacent consumers requiring separate audit

The source scan found four other scientific examples that call `program.store_history(..., depth=1)` before `program.commit(...)` and read slot 0 after `pops.run`: `euler_poisson.py` (`gradient`), `field_transport.py` (`gradient`), `variable_coefficient_field.py` (`potential`), and `api040_m08_guiding_center.py` (its stage and probe diagnostics). Their requested diagnostic semantics and clock cadence need separate evaluation; the same ring rule makes a stale final observation plausible after more than one accepted step. I made no changes to them and do not claim a reproduced failure for those examples. Tests that inspect slot 0 may intentionally test working or recycled buffers and should not be mass-edited by search-and-replace.

Evidence: `include/pops/runtime/program/program_runtime_state.hpp` (`HistoryManager::rotate`), `python/pops/runtime/_runtime_instance.py` (`history_global`), the example diff at `28f9be5`, and `docs/development/api_040/evidence/m06_history_two_step_{probe.py,receipt.json}`.
