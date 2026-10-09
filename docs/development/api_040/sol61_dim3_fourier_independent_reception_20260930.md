# Independent reception of actual Dim3 Fourier states

2026-09-30. Review checkout starts at `b2959ba2df0b88a2af2a5dfa6e6a5b2042a49da9`.
The actual installed execution source is
`48871851f10cd0acb4c0dfa45bd0f3195ab2ef0a`; execution belongs to root.
Only tests and this report/receipt are changed by this reception.

Both externally sealed profiles are received by the new autonomous verifier
`tests/review/sol61_dim3_fourier_oracle.py`. It imports NumPy and the standard
library only, with no PoPS, producer fixture, or producer mathematical oracle.
Its optional historical source check reads exact Git blobs through `git
cat-file --batch`; it never reads the changing checkout as scientific authority.

The external owner identities and manifests remain outside agent control:

| Profile | Owner identity SHA256 | Payload manifest SHA256 |
| --- | --- | --- |
| Serial | `5ff0b84abb22189d56d66dac26a00d13e0ba2fd10162652347ca2c636b86405d` | `652357acd9fef343ab0a7a9723ee34a9d520f69d66cbecf7c2f956d116882cbf` |
| MPI2 | `439694a9377920b26d63af81d36c7e1ce007b3236fd47de2b9df0538063a431f` | `5c2d35ef17db2536269b214f97eec6dc238a0b3143733fa2917a705b7f65916c` |

The saved directories are the root-owned
`outputs/coupled-gradient-dim3-sdk20d-saved-{serial,mpi2}-20260930`.
The owner identities are
`outputs/coupled-gradient-dim3-sdk20d-{serial,mpi2}-owner-identity-20260930.json`.
Full absolute paths and the independent metrics are frozen in the companion
`sol61_dim3_fourier_independent_receipt_20260930.json`.

The receipt SHA256 is
`e9a17403ffeceb072958b39e7459d62f7f581c84674c47bb607ffc6126d5e753`;
the oracle SHA256 is
`c39fcad3eddc3550c6fd2a49aab4ee8f7d6f52d4060febd3c6605482b39551c4`.

## Source and native execution authentication

The source inventory SHA256
`bd5db095e585eb43604d4da47e9c2ac3ff17e1095c6a132098e44bebae0b07d4`
names 1,109 files. All 1,109 hashes are independently compared with historical
Git blobs at the actual execution commit, together with the fixture hash
`8c0ecf72fdd67e4cf695635128476d4be909044d80a00c2f8a47b59b12b46ee3`.
The caller-pinned owner identities also bind the source inventory, build log,
launcher identity/result and each rank's JUnit file. The audit checks those
digests and their semantic links before accepting any numerical observation.

The native Dim3 DSO SHA256 is
`0ed823f5f1639172e9913a5fe5e1f15755a70a153e128e7a0f3a1274183c1bdb`.
Its ABI records native dimension 3, MPI, and SDK headers
`20d956005fe76c85c846c9ef36ddf769a80fb0d5d78d9f1e98ad220238f99e58`.
The pinned platform manifests retain the production System target and OpenMP
CPU execution. The two Serial JUnit cases pass without skip; both MPI ranks
have two passing cases with matching artifact/dimension/rank/size properties.
The MPI result authenticates before/after installation, unchanged test sources,
rank parity, and exit zero. This is reception of root's actual execution;
the agent does not launch native code or claim an independent MPI run.

## Independent scientific calculation

The physical grid has 72 spatial cells and 144 scalar unknowns, with cells
`(6,4,3)`, domain lengths `(1,2,3)`, and physical spacings `(1/6,1/2,1)`.
This distinction avoids calling 72 the total scalar degree count.

The verifier constructs the prescribed cell averages as six closed trigonometric
modes. Their exact cell-average attenuation factors are `3/pi`,
`2*sqrt(2)/pi`, and `3*sqrt(3)/(2*pi)`. The discrete eigenvalues for the pure
x/y/z modes and oblique mode are respectively `-36`, `-8`, `-3`, and `-47`.
Each two-component modal amplitude is evolved by the explicitly declared
SSPRK2 polynomial `I + dt*A + dt^2*A^2/2`, where
`A = 2*lambda*(D+R)`, `dt=1e-5`,
`D=[[.25,.05],[.05,.10]]`, `R=[[0,-.3],[.3,0]]`.
There is no FFT, rolled stencil, graph matrix, inversion, or opaque solver in
this calculation. It is distinct from the existing connectivity/Fourier checker.

The independently calculated initial and predictor fields generate every
original face flux by a physical-axis difference and explicit component
contraction. Each observed row must retain its owner-pinned operation and stage
identity, one of the two occurrences, exact oriented face measure and the
quadrature duration `dt/2 * {.5,1.5}`. All 3,456 incidences per case are unique
across the union of rank batches. Native POPSEX01 bytes are decoded separately
and compared bitwise with their JSON projections, including signed-zero doubles.
Cell amounts, actual accepted-state increments, global conservation, energy
decrease and component permutation are checked at the original fixed criteria.

Across the four actual profiles/cases (two permutations in Serial and MPI2):

| Quantity | Maximum error |
| --- | --- |
| Original SSPRK2 accepted state | `4.440892098500626e-16` |
| Independent face flux | `1.7486012637846216e-15` |
| Independent face amount | `6.564505341220828e-21` |
| Cell amount balance | `1.5299109093307173e-17` |

Every case decreases the declared quadratic energy by
`8.885562614580067e-6`. The actual native clock is exactly `1e-5` and the
accepted macrostep is the integer one.

## Counter-cases and execution scope

The new test file has 24 controls: two actual sealed receptions, twelve fully
harness-resealed artificial equations, one separate duplicate-incidence attack,
two attempts to replace the external owner pin, six resealed provenance attacks,
and one independent modal/anisotropy control. The existing 21 checker controls
also pass unchanged. All altered files live in private temporary copies; donor
payloads, external owner identities, environment and native artifacts are intact.

The artificial equations replace physical spacings with reversed axis spacings,
transpose the nonsymmetric `B`, omit the first occurrence, use the old predictor,
shift the constant initial state, or duplicate stage-0 rank contributions and
use the corresponding Euler equation. Each artificial amount ledger and final
state satisfy their own discrete equation, periodic balance and energy decrease,
within the binary64 state-subtraction bound. Their NPZ diagnostics, wire/JSON
records, receipt wire digests, saved amount arrays and payload manifest are
refreshed coherently by the harness. Both mathematical verifiers still reject
the original-equation or prescribed-initial mismatch. These copies are negative
unit evidence, never fabricated native positive results or new owner seals.

Separate resealed execution attacks prove that a failed JUnit case, foreign rank,
foreign dimension, foreign artifact, reused rank file, and a changed historical
source blob remain inadmissible even when their internal digests are updated.
Hashes cannot establish execution without the independent owner authority;
the checker deliberately preserves that boundary.

MPI2 has record partition `[3456,0]` in both permutations: one owning rank and
one empty rank. This receives all-rank execution and empty-rank participation,
but does not qualify several nonempty owners or inter-rank patch interfaces.
The fixture is one periodic Uniform grid, two components and one SSPRK2 step.
It does not receive AMR, EB, ALE, restart persistence, GPU, spatial/temporal
convergence or a complete migration PDE. A historical preparation note says
POPSEX02; the current fixture and actual sealed bytes are POPSEX01.

Reproduction, with a read-only historical Git repository and explicit inventory:

```sh
env -u PYTHONPATH POPS_DIM3_OWNER_OUTPUTS=/absolute/owner/outputs \
  /absolute/pops-api040/bin/python -m pytest -q \
  tests/review/test_sol61_dim3_fourier_oracle.py
```

The standalone CLI additionally accepts the saved directory, `--identity`,
`--identity-sha256`, `--manifest-sha256`, and optional `--repository`.
It cannot create or reseal an owner identity. Ruff and Git whitespace checks
pass. No installed package import, JIT, native build or donor mutation is used.
