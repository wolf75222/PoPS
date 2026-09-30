# Independent Dim3 saved-evidence reception

This is an offline verifier of the saved Uniform Cartesian Dim3 fixture, using
NumPy and the Python standard library only. It imports neither PoPS, the fixture,
nor any author oracle or test helper. Its unit evidence is explicitly synthetic;
no Dim3 native run or MPI2 result was available when this verifier was frozen.

## Authority and physical calculation

The fixture was reviewed at author `800fb430`, integrated as `74a47abb`, from the
independent checkout base `40e9b87f124907381e295174a8039f5365422740`.
The root corrections `c20475e9` and `def8c752` make its wire expectation
`POPSEX01`, record the actual runtime time/macro-step, and add rank/dimension
JUnit properties. The fixture file at `def8c752` has SHA-256
`8c0ecf72fdd67e4cf695635128476d4be909044d80a00c2f8a47b59b12b46ee3`.
Future receptions must pin the file actually run, independently of this note.

The prescribed domain is `(1,2,3)`, with `(6,4,3)` cells and canonical saved
storage `(component,z,y,x)`. Cell widths are `(1/6,1/2,1)`, cell volume is
`1/12`, and total volume is six. The two-component system is

```
D = [[.25, .05], [.05, .10]]
R = [[0, -.3], [.3, 0]]
B = D + R
du/dt = (.5 + 1.5) div(B grad(u))
dt = 1e-5
```

`D` is symmetric positive definite and `R` is skew. The verifier assembles the
72-DOF periodic Laplacian from explicit neighbor connectivity, with x varying
fastest; it does not copy the author's `np.roll` stencil. With component-major
flattening the evolution operator is `A = 2 kron(B,L)`. The two SSPRK2 stages are
`p=u0+dt*A*u0` and `u1=(u0+p+dt*A*p)/2`. A separate discrete Fourier calculation
uses `lambda=-4 sum(sin(pi*m/n)^2/h^2)` and amplification
`I+dt*(2 lambda B)+dt^2*(2 lambda B)^2/2`. The graph and Fourier solutions must
agree. Unit anchors for the x/y/z and oblique modes are `-36,-8,-3,-47`.
The identity `A+A.T=4 kron(D,L)` checks the skew contribution's cancellation
from the Euclidean energy form.

The initial Fourier cell means are integrated directly as
`(exp(i*k*upper)-exp(i*k*lower))/(i*k*h)`. The two physical components are
`1+.1 Re(oblique)+.03 Re(y)+.02 Im(z)` and
`-.2+.07 Im(oblique)+.02 Im(x)+.015 Re(z)`. Consequently a coherent constant
shift of all initial/final arrays cannot pass by preserving the PDE and balances.
Both component orders `01` and `10` are converted to canonical physical order;
native component `c` maps to physical component `order[c]` in the face check.

The genuine producer was checked in `prepared_diffusion.hpp`: its centered
gradient plus second-difference face correction telescopes to the two-point
positive-axis flux `(B*u_right-B*u_left)/h` for this constant tensor. The
CoupledGradient producer uses lower orientation -1 and upper orientation +1.
Each face measure is the product of the two tangent cell widths. Each occurrence
weight (.5 or 1.5) is multiplied by the SSPRK2 quadrature duration `dt/2`.
There are exactly `72*3*2*2*2*2=3456` distinct cell/axis/side/component/occurrence/
stage incidences. The verifier checks their domains and uniqueness, each raw
flux and integrated amount, each saved accumulation, cell balances, periodic
global balances, and energy decrease. It retains the author's fixed guards:
initial `2e-14`, state `4e-12`, flux `3e-12`, amount `4e-13`, energy drop `1e-9`.
No relative tolerance or guard relaxation is introduced.

## Evidence and provenance interface

`check_coupled_gradient_dim3_reception.py` accepts an evidence directory containing
exactly the two `coupled-gradient-dim3-order-01` / `-10` directories. Each must
contain `initial.npz`, `accepted.npz`, `increments.npz`, `native-ledger.json`,
`receipt.json`, and one `ledger-rank-NNNN.bin` for each rank. Unexpected files,
symlinks, missing/duplicate NPZ fields and nonfinite numbers are refused. Rank
profiles are bounded to serial or MPI2. JSON duplicate keys are refused.

The manifest contract is `pops.coupled-gradient-dim3.offline-manifest@1`.
The separately pinned identity contract is
`pops.coupled-gradient-dim3.external-identity@1`. A metadata draft can be emitted:

```sh
python check_coupled_gradient_dim3_reception.py EVIDENCE --draft-identity > identity-draft.json
```

Its `pending-owner-pin` evidence kind is deliberately unverifiable. The native
campaign owner must authenticate and freeze the actual native extension digest,
ABI, package/native paths, fixture digest, dimension/rank count, artifact token,
platform manifest, operation identity and both exact StagePoint evaluation
identities in `cases.01` and `cases.10`. For `evidence_kind=native-reception`, add
`native_source_commit` (40 lowercase hex characters), `native_exit_status=0`, and
`junit_by_rank`: one distinct entry per rank with `rank`, absolute `path`,
`sha256`, and `testcases` mapping orders `01` and `10` to their exact pytest
JUnit names. Each linked case must pass, with the correct artifact, receipt,
dimension, rank and size properties. Conflicting duplicate properties fail.
Keep this identity outside the evidence directory so it is not a self-issued
payload receipt.

```sh
python check_coupled_gradient_dim3_reception.py EVIDENCE --identity IDENTITY --seal
python check_coupled_gradient_dim3_reception.py EVIDENCE --identity IDENTITY \
  --identity-sha256 CALLER_PINNED_IDENTITY_SHA256 \
  --manifest-sha256 CALLER_PINNED_MANIFEST_SHA256
```

Sealing only fingerprints existing files and prints the two digests; it never
creates native results. Verification requires both caller-pinned digests. Each
native wire is decoded independently and compared bit-for-bit with its JSON
projection, including signed zero. The receipt's wire hashes, rank counts,
runtime time `1e-5` and macro-step one are mandatory. Saved author predictor/final
diagnostics are cross-checked against the independent solution, never used as an
authority. `POPSEX02` can also be decoded for consistency, but this static graph
has no integral or consumed trace section; the current actual fixture emits v1.

## Independent injections and validation

`tests/review/test_dim3_saved_reception_checker.py` exercises the standalone
verifier without importing PoPS. Temporary control data are labeled
`synthetic-unit` and use conspicuous synthetic identities. They are verifier unit
inputs, never native scientific evidence or replacement runtime outputs.

The 21 cases include the analytic cell-mean/mode anchors and two-permutation
control; unrescaled payload corruption; foreign native identity; JSON/wire
disagreement; extra payload; absent clock; failed/skipped/spoofed-rank/foreign-
artifact JUnit links; JSON/NPZ nonfinitude; an unauthorized draft and an altered
external identity despite resealing the manifest. Six scientific attacks are
fully resealed and reach equation guards: changed final state with consistent
state increments/amounts; coherent initial shift; wrong raw face flux; wrong
quadrature duration; incorrect component permutation; duplicated incidence.
For ledger attacks, wire, JSON, integrated amounts, receipt wire hashes, saved
face increments and manifest are updated together. Their refusal therefore
does not depend on an uncorrected checksum.

Validation command (source/host only):

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  -m pytest -q tests/review/test_dim3_saved_reception_checker.py
```

Frozen source/host reception: **21 passed in 3.12s**, Ruff passed, and staged
`git diff --check` passed. The central owner must
still run the authentic rebuilt Dim3 package for serial and MPI2, pin its build /
launch / JUnit provenance, and pass the checker on those actual files. This
bounded verifier does not qualify AMR, EB, GPU, restart/rollback, moving geometry,
other grids/integrators, or the compiled execution path. Saved files and JUnit
cannot cryptographically prove execution: their external build and launch pins
remain a campaign-owner obligation.
