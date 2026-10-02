# Explicit M16 Program call — Native fixture preparation @1

Author base `e943942c6a5eef3c9172a8a5dcffd766e1510c4f`, preserving sharedbasis
`78d9624` and implementation `251c510`. SOURCE / CPU header preparation only.

The dedicated installed-package fixture is
`tests/python/integration/runtime/test_program_affine_moment_explicit_basis_runtime.py::test_installed_program_explicit_monomial_binding`.
It exercises validate → resolve → genuine compile → bind → run, including the new
Program affine call's explicit basis and State mapping. Basis indices are reordered
independently of the 21 arbitrarily named State slots. Degree5, dimension2, synchronous
single-level AMR, periodic4×4, zero physical transport. The first-moment endpoint is
the actual DenseLU solve of the linear rotation equation; all moments use the same
map. The independent three-atom rational oracle uses a=1/3, c=4/5, s=3/5. The represented
binary64 Omega differs from its rational ideal by rounding; the success bounds are
rtol4e-12/atol2e-13. This is a deliberately small polynomial witness, not a HyQMOM15
hyperbolicity, full M16, AMR reflux or arbitrary-degree performance receipt.

The nonfinite case has finite initial data and produces an infinite endpoint in a
real compiled expression. The overflow case has finite initial highest-degree moments;
its actual public-header probe confirms the polynomial refuses output. Both Native
runs require RuntimeError/nonfinite rejection on every rank, byte-exact valid arrays,
complete carrier bytes, and unchanged time/macrostep/level count. No failed bind is
accepted as a successful rollback witness. Snapshots and failure triples are persisted
before assertions. Persistence is itself under a collective boundary. The receipt
pins actual installed Python origin, selected DSO bytes, before/after NPY and complete
carriers. It is test evidence, not a ROOT approval/seal or independent scientific receipt.

Serial and MPI2 use the same genuine API. MPI compilation reuses the repository's
compile-once helper; native operations and local assertions use the collective helpers.
The World collective directory supplies the same rank0 capture destination. This
preparation has not executed Native, MPI, Kokkos/device, or JIT. ROOT must authenticate
its actual installed SDK/origins and run the exact nodes after its rebuild.

Suggested ROOT commands from this checkout, with SDK_PYTHON and SDK_MPIEXEC pointing
to the actual authenticated installed interpreter and matching MPI launcher:

```sh
env -u PYTHONPATH "$SDK_PYTHON" -m pytest -q -o pythonpath='' tests/python/integration/runtime/test_program_affine_moment_explicit_basis_runtime.py --junitxml="$OUT/serial.xml" --basetemp="$OUT/serial-tmp"
env -u PYTHONPATH "$SDK_MPIEXEC" -n 2 "$SDK_PYTHON" -m pytest -q -o pythonpath='' tests/python/integration/runtime/test_program_affine_moment_explicit_basis_runtime.py --junitxml="$OUT/mpi-rank.xml" --basetemp="$OUT/mpi-tmp"
```

MPI JUnit/basetemp should be rank-separated by ROOT's existing launcher receipt
convention; the above spells the genuine command payload, not a substitute harness.
No source package may shadow the installed PoPS. Preserve before/after installed
identity receipts and all rejection logs.

SOURCE reproduction:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest -q -o pythonpath='python .' tests/review/test_sol61_explicit_affine_runtime_preparation.py
```

Result: 5 PASS12.19s, including public validate/resolve for all three cases, Source
origin/_pops-absence, independent atomic first moments, and actual complete-header
CPU overflow refusal with unchanged output. No Native result is inferred.
