# Original Uniform Fan–Li15 eight-step fixture

The new fixture contract `pops.fan-li15-public-composition-native-fixture@2` retains the original N=16, dt=1e-4 and eight SSPRK2 temporal steps. Fifteen components mean ten conservative equations plus five terminal nonconservative rows. Canonical and reversed multi-index orders are separate parametrized runs. This is the existing public normalized-analytic path realization; the original script's Gauss4 default and every original numerical criterion remain unchanged.

The fixture performs public validate → resolve → compile → bind → run. Resolve requires actual model source retention on every model. Before bind, it persists the verified exact Uniform layout/program identity, complete compiled manifest, actual Model C25 evidence and source dumps, Program IR/C++ dumps, and Model/Program/native DSO hashes. Source files and binary paths describe genuine artifacts; no reader seals or scientific approval are generated.

Each of nine phases (initial, accepted1 through accepted8) persists its valid NumPy array and clock immediately. Attempt and capture failures are retained without replacing the original collective exception. A rejected run is labelled rejected-attempt, never accepted. The fixture explicitly reports Uniform full-storage carriers unavailable: these captures do not qualify grown ghosts, checkpoints, restart or full-storage rollback. It neither substitutes AMR nor calls private observation APIs.

Scientific checks retain original initial/state/time/y-invariance/inventory criteria, positive density and covariance, active third/fourth moments and nonconservative terms. A separate vectorized signed FV/SSPRK2 reference is checked at every phase; the original independent DOP853 semidiscrete reference is checked at eight steps. The independent reception reader is authored separately by Banach; fixture success does not grant ROOT approval or full M17 qualification (other truncations, discontinuities and AMR remain separate).

Source preparation command (no native module/build):

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider -o 'pythonpath=python .' tests/review/test_sol61_m17_full_eight_preparation.py
```

Future installed-package nodes, executed only by ROOT with authenticated SDK/compiler/Kokkos environment and the repository installed-check runner:

```text
tests/python/integration/runtime/test_fan_li15_full_eight_step_public_runtime.py::test_installed_original_fan_li_eight_steps[canonical]
tests/python/integration/runtime/test_fan_li15_full_eight_step_public_runtime.py::test_installed_original_fan_li_eight_steps[reverse]
```

A bounded Serial command, after ROOT has pinned the installed SDK and its required compiler/Kokkos environment, is:

```sh
env -u PYTHONPATH "$SDK_PYTHON" docs/development/api_040/run_installed_checks.py --output "$FRESH_OUTPUT" --test tests/python/integration/runtime/test_fan_li15_full_eight_step_public_runtime.py
```

`SDK_PYTHON` and `FRESH_OUTPUT` designate the authenticated installed interpreter and a new receipt directory; they are not source-import substitutions.

Both nodes must also be received independently under MPI2 with every rank participating in genuine collectives. Source preparation and collection alone are not Native execution. Historical two-step fixture and receipts remain intact. The preparatory RED about literal C++ diagnostic labels was a Source-test assumption, not a numerical or emission defect; the corrected test requires the genuine Uniform installation entry instead.
