# Accepted Halo extent: actual pybind interoperability

Hooke / GPT-6.1 Sol; parent 0e6de1b6. The authentic SDK8 representative failed at
bind before Halo execution. `accepted_halo_extent` used `def_readwrite` and asked
Python for an unregistered `pops::Extent<2>`. The Python lowering's ranked tuple
was correct; this Native property bypassed the existing conversion seam.

Minimal correction: use the same `ranked_extent_to_python` and
`ranked_extent_from_python<kNativeDimension>` property adapters as `shape` and
other ranked configuration. No Python Extent constructor exists. The converter
requires exact tuple/rank/exact integer entries (rejecting bool), positive values,
and lossless int64 conversion before assignment. Getter returns a ranked tuple.
No policy, PDE, threshold, ghost guard, model, ABI number or cache change.

Actual installed pre-fix test, no JIT/mesh/runtime construction:

```sh
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-halo9/bin/python -m pytest --noconftest -o pythonpath= tests/python/integration/amr/test_accepted_halo_extent_binding.py -q --tb=short
```

**1 FAIL in 0.18 s**, at real assignment `(1, 2)` with the same `pops::Extent<2>`
TypeError. Installed module authenticated by selector:

`/Users/romaindespoulain/miniforge3/envs/pops-api040-halo9/lib/python3.12/site-packages/pops/_native/dim2/_pops.cpython-312-darwin.so`

SHA256 `33eedc3d3db2021a092f02e7e02ef65c90b8f01c9b8ac001aa0c57da8ae5a002`.
Package likewise came from that environment. The same installed converter via
`AmrSystemConfig.shape` passed tuple roundtrip and six malformed-input refusals.
This checks the actual reusable mechanism, not the unbuilt correction.

Source regressions:

```sh
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops/bin/python -m pytest --noconftest -o pythonpath=python tests/python/unit/runtime/test_amr_bind_lowering.py tests/python/unit/amr/test_accepted_halo_preparation.py -q --tb=short
```

**43 PASS in 11.59 s**. These tests do not establish real setter compatibility;
the new interoperability test deliberately exercises the genuine pybind setter
and verifies seven refusal cases leave its prior anisotropic value intact.

Root must rebuild and rerun that focused test before repeating a representative.
No Root checkout, ENV, installed DSO or cache was modified; no compiler/JIT or
native campaign rerun. Corrected setter success and scientific Halo remain unproven.
