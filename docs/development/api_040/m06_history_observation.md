# M06: read the latest accepted history sample

The retained M06 receipt at
`outputs/m06-eac92bb-openmp1-reviewed/states/result.json` reports exact final
enthalpy but one-step-old temperature and phase fraction in all eight
scenarios. The mathematical body constructs both from the candidate `H`, then
stores them before committing `H`. The generated Program rotates the history
ring after the commit. `HistoryManager::rotate(clock)` swaps the just-written
working slot 0 into slot 1 and recycles a prior buffer into slot 0. M06
declares `depth=1`, so its physical ring has exactly two slots. The final
`history_global(name, 0)` read in M06 was observing the recycled prior sample;
the latest accepted sample is `history_global(name, 1)`.

The authentic installed `6b5f452e432976a94b69635b96aad1a04c42c8d5e933cad5cf785301089488fb`
Dim=2 native extension and
`dfcd85eba73c963aaceeaf85738296e5a46272e3f76aff1890a2f7850c4b94d8`
SDK were probed without reinstalling or
rebuilding PoPS. The [two-step probe](evidence/m06_history_two_step_probe.py)
compiles one public 4×4 M06 case and rebinds two input rates. Its
[receipt](evidence/m06_history_two_step_receipt.json) records after step 2:

| Trajectory | Published H | Slot 0 | Slot 1 |
| --- | ---: | ---: | ---: |
| Sensible, temperature | 1.5625 | 0.765625 (step 1) | 0.78125 (step 2) |
| Plateau, liquid fraction | 2.5 | 0 (step 1) | 1/6 (step 2) |

The probe checks the complete 4×4 arrays, not only the displayed cell. At
step 1, cold-start filling makes both slots equal, which explains why a
single-step check would not discriminate this error. The native SHA and ABI
are recorded in the receipt. Its assertion passes for both trajectories.

Only M06's final observation now reads slot 1. Its update equation, symbolic
thermometer, guards, time step, initial conditions and tolerances are
unchanged. The full eight-scenario example has not yet been rerun after this
source fix; that remains the installed acceptance check. This is an
observation-index correction, not a change to the history rotation contract
or a codegen fix.
