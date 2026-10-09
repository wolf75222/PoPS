# Installed runner canonical origins

The identity writer now stores the resolved package-file origin, matching the MPI worker's existing canonical-path convention. Identity schema2 is unchanged: this repairs inconsistent representation of the same field, not its authority. Distinct canonical files remain rejected even when their bytes match. Native hash and execution-environment comparisons remain strict.

Each worker persists its own observed identity and a schema1 `rankN.identity-check.json` diagnostic containing all gathered identities and exact expected/observed mismatches before raising an authentication refusal. No observed result is substituted into the expected identity. Missing rank XML now makes test parity false, including when all rank XML files are absent.

Source validation: four tests cover a real filesystem symlink versus a different same-content file, hash/environment mismatches, missing-XML parity, and persistence ordering. Command: `env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 <pops-python> -m pytest --noconftest -p no:cacheprovider tests/review/test_installed_runner_origin.py -q` (4 PASS, 0.09s). These are runner-only checks; no Native/MPI execution or scientific reception is claimed. Failed job731805 remains historical evidence.
