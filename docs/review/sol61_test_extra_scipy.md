# Scientific test dependency: SciPy

Actual SDK20 ROMEO job 732072 reached eight returned steps and nine saved Uniform Fan–Li storage phases, then failed in `test_fan_li15_full_eight_storage_runtime.py` → `api040_m17_oracle.solve_reference` with `ModuleNotFoundError: scipy`. The installed SDK20 test extra contained pytest, h5py and mpmath; SciPy was absent. This preserved failure is not a scientific reception.

Declare SciPy in `project.optional-dependencies.test`, the dependency inventory installed by the SDK build. It is needed by the existing independent `scipy.integrate.solve_ivp(..., method="DOP853")` oracle. No new minimum version is imposed; the dependency resolver chooses a release compatible with Python and NumPy. Core dependencies, equations, numerical guards, fixtures and C++ are unchanged.

Validation is limited to TOML parsing and an actual installed local SciPy DOP853 finite-result check, without PoPS import or Native execution. No ROMEO environment is modified here, and the received 428-PASS Source cohort and SDK20 RED remain immutable. ROOT owns future provisioning and authentic rerun.
