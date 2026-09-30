# Typed initial storage through composed reference geometry

The first real SDK20d public ALE campaign fails all eight pytest cases before
the native problem artifact is loaded. Its typed BindArray preflight reads
only layout.mesh/n, while MovingControlVolumes publishes the same exact
reference geometry through normalized_geometry(). It therefore refuses a
complete declared initial array despite the resolved moving provider.

The authentic red receipt is in the task workspace at
outputs/installed-public-ale-saved-states-dim1-sdk20d-20260930. Its identity SHA
is 81cd4e05883a3a2364d71f035a888f267db42d744148904f0b25370b2e517922;
log SHA is 047f2448e3c44f6479da7bccc170699a0c810efaac1f19b4ebc5127c2463640e.
The installed Dim1 native SHA is
84eefc63aabfc4b643f27f51d24a1fa5ec202db2c722c7e10f6c91f84ea2fd40,
SDK20d, ABI5. This red campaign is retained; it contains no accepted ALE states.

The bind gate now reads the exact NormalizedGeometry reference-cell contract
for descriptors that expose it. It requires the nominal immutable type and
refuses a duck-typed projection instead of falling back to loose mesh metadata.
The existing AMR runtime-layout query and legacy descriptor fallback remain.
Initial storage still requires the complete component-major array, reversed
native axis order and the declared dtype. Reference counts do not authorize
evolved cell measures: the moving geometry/provider keeps its own point,
metric, generation and publication guards. No Cartesian placeholder, model
name dispatch, ABI layout change or numerical guard relaxation is introduced.
This is a repair of consuming the existing geometry contract, not a new wire
or API contract version.

The coherent source gate suite receives51 tests, zero skips/failures: the
existing bind-validation suite plus exact1D/2D/3D axis order, incomplete arrays,
wrong dtype, invalid geometry projection and a genuine declared moving layout.
These are pure source tests, not native ALE execution. Ruff and diff checks
pass. Independent review, installed-package rebuild and native serial/MPI2
reception are pending.

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q --tb=short -o pythonpath=python tests/python/unit/runtime/test_initial_reference_geometry.py tests/python/unit/runtime/test_bind_validation.py
```
# Subsequent installed reception and remaining refusal

The corrected SDK20d Dim1 campaign at source `470dfb08774876aff411202c12491a02bf453197`
is retained in `outputs/installed-public-ale-initial-reference-corrected-dim1-sdk20d-20260930`
in the task workspace. It runs the genuine installed package: six cases pass,
two vector-with-source cases fail, and none skip. The remaining refusal occurs
after accepted runtime steps, during moving output checkpoint capture:
`receipt quantities/support differ from accepted exchanges`. This is a separate
native receipt validation failure; it does not qualify the complete ALE chain.
The identity SHA is `94833e0ad6785b0dc93d682c069ec73d081afc0ad4f236627918fc03a07eeb38`
and log SHA is `9183a4dfd8359fab77c881bf9d61be5b429666634783a2b42edd0f84ecae53cc`.

The checkpoint mismatch diagnostic now retains the exact operation/occurrence
and hexadecimal actual/expected quantities. Equality and support guards remain
unchanged. A rebuilt native package is required to use this diagnostic; this
change alone is not a repair or a successful scientific reception.
