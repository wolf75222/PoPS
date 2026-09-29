# Affine AMR fixture: require a real coarse/fine interface

The N=8 and then N=16 fixtures selected the high-density half of the periodic
box using `rho > 1.3`. Actual native binds produced complete fine coverage,
including **1024/1024** fine cells at N=16. Increasing N alone did not qualify a
coarse/fine interface. The existing strict partial-coverage assertions correctly
rejected those attempts; their native failure receipts remain unchanged.

The geometry selection is now the public Boolean tag
`(rho > lower) & ~(rho > upper)`, with explicit RuntimeParam defaults
`lower=1.25` and `upper=1.35`. For the original linear density
`rho(x)=1+.6*x`, this selects the two interior coarse columns 7 and 8 at N=16.
It does not select cells next to the periodic x seam. Native nesting and
clustering retain their ordinary stencil requirements. `PatchLayout` governs
coarse patch distribution and does not offer an explicit fine-patch authority,
so the fixture uses the existing public tagging protocol.

The old threshold touches the periodic seam; the real hierarchy's periodic
nesting/clustering is sensitive to that geometry. The repair does not assert a
defect in that machinery or prescribe its internal patch decomposition. It
selects an interior region and still checks the actual resulting coverage.

The physical density, exact FV cell means, particle weights/velocities, moment
multi-indices, affine map, LocalResidual, seed, dt=.01, and **3e-12** comparison
tolerance are unchanged. No post-update state correction, projection, clamp,
or replacement of the independent particle oracle is introduced. Changing
tagged geometry is a sampling choice for this local-operator witness, not a
change to its mathematical problem.

## Actual installed reception

The existing two native tests ran from the isolated checkout based on `c6625681`,
using installed PoPS and an isolated cache, one compiler at a time:

- `test_public_affine_body_and_local_solve_match_particles_on_amr[2]`: initial
  exact means and accepted affine image match the particle oracle on valid
  cells of both levels; one accepted step and correct time/macro-step.
- `test_impossible_local_residual_rejects_without_amr_publication`: a real
  partial refined hierarchy, two rejected attempts, bitwise unchanged states,
  unchanged patch boxes, levels, clock and macro-step.

Result: **2 passed / 2, no skips, 54.7596 s**. Both existing assertions
`0 < fine_valid_cells < fine_domain_cells` executed and passed. A subsequent
bind reused the same isolated artifact cache and recorded **512/1024** valid
fine cells, with fine box `[8,0]..[23,31]`. This measurement is evidence, not an
added requirement that the implementation must always choose that exact box.

Receipt directory (workspace output): `outputs/astra-affine-amr-interior`,
including `identity.json`, `result.json`, `pytest.xml`, `pytest.log`,
`source-files.json` and `geometry.json`. Native DSO SHA256 is
`aed8c1582c3344bd5afb19ab784adec260bcdbd4bf944b26a29242b53889c862`;
SDK signature is `396719235acdb96435dba0dd27c5e5579aefbda4936f657262b9b2553d13bb4c`.
The runner verifies every installed production file against the checkout and
authenticates native Dim2 inside the pytest child. This is a serial run of the
MPI-enabled CPU backend; MPI execution and GPU remain unqualified here.

Reproduce using the repository's installed runner, with PYTHONPATH unset,
OMP_NUM_THREADS=1, Kokkos_ROOT pointing to the existing environment and
POPS_NATIVE_DIM=2:

```text
python docs/development/api_040/run_installed_checks.py --output OUTPUT \
  --test 'tests/python/integration/runtime/test_affine_push_forward_amr_runtime.py::test_public_affine_body_and_local_solve_match_particles_on_amr[2]' \
  --test tests/python/integration/runtime/test_affine_push_forward_amr_runtime.py::test_impossible_local_residual_rejects_without_amr_publication
```
