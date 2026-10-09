# Independent reception of the M18 metadata correction

Target: `d6ec6706895c39f2a0d6ad7f9ae5332df16cab83`, parent exact
`81249ef170bba6e8176b3ff54b42e58ae275e24d`.
Helper SHA256: `36027f5658a8d1a364681e89cc1bb024c798bbed4e19cb0de49f4fbcdd53068e`.
Historical counterexamples remain frozen in
`5cbcdf4c5ada9627305d6020e629fb9daccd65f1`; neither their source nor their
accepted-counterexample expectations were edited.

The new probes execute the correction's exact Git-object helper and reuse
the historical protocol fixture from its exact Git object. All temporary
files contain opaque bytes explicitly labelled as protocol-only. The probes
never manufacture accepted scientific states or external owner approval.
They do not execute PoPS native code, MPI, a compiler, JIT or installation.
No production or receipt donor was modified.

## Received refusals before publication

The same native/C++ alias attacks accepted by helper81249 now refuse before
the sidecar exists. The independent reception also covers Python, SDK, dual
and target symbolic origins, parent-directory aliases, dangling C++ aliases
and `..` paths. Actual diagnostics are
`M18 execution origin path aliases are forbidden`.

Duplicating an entry within one Program's original generated-source list
refuses both with a present file and with a missing file, before union or
the permitted missing-metadata fallback. Diagnostic:
`M18 Program generated source inventory contains duplicates`.
The historical helper still accepts those exact inputs in its separate
paired control; it is not requalified as corrected.

Changing each Python/SDK/native/dual/target file after initial consensus now
refuses with `M18 execution origins changed before recording` before opening
the sidecar. Switching to a different provider path with identical bytes,
changing the observed fixture HEAD, and removing local C++ metadata are also
refused instead of becoming newly authenticated authority. Elected C++ drift
retains `M18 elected generated source changed before recording`.

Three additional tests inject native-byte drift specifically at the
destination-path vote, once for each simulated rank0/1/2. Revalidation's
failure vote is reached; the final agreed-record vote and file opening are
not reached. These are bounded collective-protocol simulations, not actual
MPI runs or distributed ownership qualification.

## Received positive contracts and order

For each of three simulated ranks, observable order is:

```
capture original record
allgather rank records
allgather destination path
reopen elected C++ and recapture complete local record
allgather final agreed record
rank0 exclusively opens sidecar (x)
all ranks enter final exception-convergence vote
```

Non-writing ranks still complete final convergence. Repeated exact C++ paths
shared by two distinct Programs are admissible; duplicates inside one Program
are not. A peer with no local generated-source metadata reopens the elected
compiler-owner source and retains its own original `null` local record during
recapture. Actual missing metadata remains `null`, with no sibling discovery.
The unmodified original schema and file digests match helper81249 on ordinary
nonaliased inputs. The sidecar still does not constitute ROOT approval, build
source authority, expression mapping or a native scientific result.

No remaining defect in the three reviewed P2 seams was demonstrated. Hashing
does not make independently mutable files atomically immutable: external
ROOT approval, strict assembler checks and authentic native evidence remain
necessary. This receipt does not qualify a future M18 Serial/MPI2 execution.

## Actual source/host commands

From private checkout `PoPS-sol61-m18-owner-fixed-reception`:

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -c 'import sys; sys.path.insert(0,"python"); import pytest; raise SystemExit(pytest.main(["-q","-p","no:cacheprovider","tests/python/unit/runtime/test_m18_execution_owner_capture.py","tests/review/test_sol61_m18_execution_owner_fixed_reception.py","tests/review/test_sol61_m18_owner_assemble.py"]))'
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check tests/review/test_sol61_m18_execution_owner_fixed_reception.py
rtk git diff --cached --check
```

Actual result: **111 PASS in 2.72 s**: twenty-eight independent checks,
forty author capture checks and forty-three assembler checks. Ruff PASS.
The earlier standalone run had twenty-five PASS before adding the three
post-path-vote drift tests. Only an unused test import was removed afterwards;
no refusal or scientific guard was weakened.
