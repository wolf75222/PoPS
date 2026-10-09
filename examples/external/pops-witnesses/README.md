# External formula witnesses

This independently installed package exercises the public PoPS composition API
at core revision **620ac6b6d2ad532a7616652d1bdaac9ca507287a** (#681).
The contribution lives outside the PoPS Python package, headers, bindings and
runtime. It adds no operation, compiler transformation or native kernel to PoPS.

## Two changes to the same composition

The periodic rectangle is [0,2] x [0,3], with 16 x 12 cell means, binary64,
h=1e-4, one accepted SSPRK3 step and an unchanged donor d. At each of the three
stages, existing CG solves (-Delta+3)phi=q+d and (-Delta+5)psi=q-3d/4.
Each solve is consumed with FailRun, observed and published for its own stage.
Diffusion uses the existing CoeffGradient/div operations with coefficient 0.15.

| Variant | epsilon | beta | Local source |
| --- | --- | --- | --- |
| baseline | 0 | 0 | 0.2 phi + 0.4 psi - 0.8 q |
| linear / linear-renamed / beta-zero | 1/20 | 0 | 0.25 phi + 0.35 psi - 0.8 q |
| cubic / cubic-renamed | 1/20 | 1/20 | 0.25 phi + 0.35 psi - 0.8 q - 0.05 q q q |

The cube is **the cube of the stored cell mean**, (q_bar)^3. This differs from
the cell average of the continuous q^3. Acceptance establishes this discrete
composition; it does not establish continuous PDE accuracy, convergence order,
global nonlinear stability or generic support for every time method.

[author.py](src/pops_witnesses/author.py) contains the physical declarations,
using Model/source/rate, FieldProblem, solve/observe/publish, Program values,
histories and commit_many. It has no imports from PoPS internals or repository
examples. Renaming owners, states, fields and publications must leave every
canonical output byte identical. beta-zero explicitly recovers the linear
witness.

## Independent references and acceptance

[reference.py](src/pops_witnesses/reference.py) imports neither PoPS nor the
author. Its initial arrays come from independent cell-face integration.
A scalar periodic five-point stencil and full-grid FFT screened solves advance
the SSPRK3 recurrence. Tests cross-check the fields with an independently
assembled dense matrix; linear trajectories also have a scalar affine modal
reference. A single-mode nonlinear oracle would discard generated harmonics.

The budgets are fixed from CG controls (max_iter=2000, rtol=1e-12, atol=1e-14),
positive screenings, binary64 roundoff, a joint RHS ceiling of 4 and Q=3.
The nonlinear Lipschitz bound includes 3 beta Q^2. Every initial, intermediate
and accepted q must stay finite inside Q=3; exceeding it invalidates the budget.
Independent saved-field residuals and all twelve saved arrays are checked.
The donor must be byte identical to its actual bound initial value.

The first-stage changes must satisfy, cell by cell:

- linear-minus-baseline: h epsilon (phi0 - psi0);
- cubic-minus-linear: -h beta q0^3.

For the cubic change, the (2,0) coefficient is measured as
2/(Nx Ny) sum(delta_q1 cos(2 theta_x)), summed over both spatial axes,
theta_x=2 pi (x+1/2)/Nx. With A=(1/8)sinc(1/Nx)sinc(1/Ny), its independent
analytic value is -h beta (3*2*A^2/4), approximately -1.13e-7.
This number is a prediction until an execution receipt reports an observation.
The final-state changes and this coefficient must agree with the reference
within the frozen budgets and exceed 100 times their comparison budgets.

Negative tests reject unchanged baseline outputs, q^2, a linearized source,
fields frozen at stage zero, enclosure violations and a one-bit donor change.

## Run against an installed native build

Build/install the frozen core before installing this separate package.
The recorded campaign uses ROMEO x64cpu SLURM jobs, Python 3.12.5,
Spack GCC 14.2.0 and privately installed Kokkos 4.4.01 Serial.
The Kokkos release tarball SHA256 is the one in the frozen CMakeLists.txt.
PoPS is built through its existing scikit-build/CMake entry point, Dim2,
POPS_USE_MPI=OFF, POPS_USE_HDF5=OFF, heavy TU pool 3; no core source is edited.
The wheel, compile_commands.json, CMake cache and build logs are retained.

On ROMEO load the same Spack Python/GCC specs, activate the private venv and
source its recorded build-env.sh. Put the compiler's libstdc++ directory first
in LD_LIBRARY_PATH before importing NumPy/PoPS; otherwise an older system
libstdc++ can be loaded before the GCC14 core. This changes the job environment.

From a compatible Linux compute job, with SOURCE pointing to the frozen core:

~~~bash
python -m pip install --no-deps -e "$SOURCE/examples/external/pops-witnesses"
python -m ruff check --config "$SOURCE/pyproject.toml" "$SOURCE/examples/external/pops-witnesses"
python -m ruff format --check --config "$SOURCE/pyproject.toml" "$SOURCE/examples/external/pops-witnesses"
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest "$SOURCE/examples/external/pops-witnesses/tests" -q

# Use a NEW empty OUT each time; preserve failures and original outputs.
for variant in baseline linear linear-renamed cubic cubic-renamed beta-zero; do
  env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python -m pops_witnesses.capture --variant "$variant" \
      --source "$SOURCE" --wheel "$WHEEL" --out "$OUT/$variant"
done

# Separate process: inspect original execution evidence before comparing arrays.
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 \
  python -m pops_witnesses.check --campaign "$OUT" --report "$OUT/check-report.json"
~~~

[RESULTS.md](RESULTS.md) records actual jobs, failures and observations.
Earlier macOS/other-revision builds are historical evidence only. They do not
satisfy either witness on this frozen core. No Windows build or calculation is
part of this campaign; WSL holds the contribution checkout and publishes Git.

## Retained execution evidence

[capture.py](src/pops_witnesses/capture.py) is the producer. It runs the public
validate -> resolve -> compile -> bind -> run path, saves actual bind inputs,
public state/history arrays and an accepted checkpoint, and logs its PID,
SLURM job and input/output identities before and after the native step.

The external [compiler relay](src/pops_witnesses/compiler_relay.py), selected
through the existing POPS_CXX override, delegates to the exact baked compiler
without changing source, argv, stdout/stderr or exit status. It preserves the
temporary TU **actually passed to the compiler**, neighboring generated
headers, dependency files, invocation and output DSO bytes before PoPS removes
temporary paths. This is stronger evidence than a later dump_cpp regeneration.
Friendly dumps are also retained, separately labeled.

ROMEO GPFS rejected the core checkpoint's rename-no-replace operation. The
producer writes the public checkpoint on node-local /tmp and copies its entire
original directory into the evidence before process exit. This changes only
the storage location; failures and staging files on GPFS are preserved.

PoPS loads model DSOs through sealed private images in /tmp/pops-native-*.
The capture identifies the actually mapped image by its bytes and checks their
equality to the declared artifact and compiler output; the catalogue path alone
is insufficient. Raw loader mappings and both paths are retained.

The producer observes loaded original paths in /proc/self/maps, copies the core,
model and Program DSOs, hashes the loaded support-library closure and checks
the SDK/installed Python/native/wheel/source fingerprints before and after.
The source bytes are compared to Git objects at the frozen revision; all selected
core files, including build definitions, are covered. Each variant uses fresh
caches. Changed formulas must change compiler inputs and loaded non-core DSOs,
while all core fingerprints stay identical.

[check.py](src/pops_witnesses/check.py) first checks sealed original files against
the execution journal, then verifies that loaded generated DSO hashes match
successful compiler outputs, and only then invokes the independent math reader.
A test demonstrates that replacement oracle arrays can pass the math comparison
and still fail their link to the original execution record.

The hashes and read-only file modes are integrity controls relative to a trusted
record. They do not prove origin against an author able to alter outputs, logs
and hashes together. A separate reviewer should obtain originals through the
cluster job record, inspect the compiler inputs and loader identities, and rerun
the reader. No compiler-to-binary formal graph proof or third-party acceptance
is claimed.

## Interfaces, native possibilities and missing contracts

| Limit | Interface usable now | Native extension possible | Contract actually absent from this witness path |
| --- | --- | --- | --- |
| Local cubic source | Symbolic multiplication and Model.source/rate already suffice | Existing native component mechanism for a different primitive, if needed | None for q*q*q |
| Stage-dependent linear Fields | Public solve/consume/observe/publish and CG on periodic SPD screened equations | Another solver component implementing the existing field interface | A generic nonlinear field-solve contract is not supplied by these linear CG declarations |
| Stage observations | Public histories preserve all three q/phi/psi samples; accepted checkpoint is saved | Expose existing native auxiliary/solver status through diagnostics | Persistent public successful-CG iteration report and publication-stage timestamp readout from accepted checkpoints |
| Formula-to-loaded-code evidence | Compiler override, compile dumps, artifact paths and Linux loader maps | A core-independent relay already records actual compiler bytes | No public end-to-end attestation tying arbitrary output bytes to execution against an adversarial producer |
| CPU Dim2, serial world | Existing installed specialization and public bind/run | Existing other native backend/dimension builds can run separate campaigns | No missing API inferred from an untested MPI/GPU/AMR configuration |
| Other temporal/multi-owner compositions | Public Program operations remain available | Further native implementations must obey ownership and stage contracts | This frozen-donor SSPRK3 receipt does not qualify evolving joint owners or another integrator |

Library changes are the source formula, parameter bounds and external proof
harness. The core delta is zero. A future need for new primitive/solver/runtime
contracts must be assessed separately, rather than hidden in this library.
