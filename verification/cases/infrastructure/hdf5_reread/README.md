# IF-10 — HDF5-shaped npz round-trip

Infrastructure identity. Serialize a TR-01 HDF5-shaped state
(centers, q, components=["q"], owner="rank0") to numpy `.npz` and reload.
Array identity after the round-trip. No h5py. `run_native` inspects the
TR-01 Case and raises `NativeUnavailable`: `Case.blocks()` exposes no
public state Handle for `ScientificOutput.fields`, and `read_hdf5` is
not a `{centers, q, components, owner}` reread.

| Field | Content |
|---|---|
| Identifier | `IF-10` |
| `verification_kind` | `infrastructure` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + a\partial_x q = 0\). Conservative scalar \(q\). Same IC as TR-01: \(q=q_0+\varepsilon\sin(2\pi k x)\) at \(t=0\). |
| Oracle | TR-01 HDF5-shaped state `{centers, q, components=["q"], owner="rank0"}` via `load_sibling_module` on `advection_sine/exact.py`. numpy `.npz` dump/load must recover the arrays with \(L^\infty=0\) and preserve component order and owner. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d periodic unit interval \([0,1]\). Samples are interior cell centers. |
| Parameters | \(n=32\) uniform cells. Dimensionless. Single-rank owner `rank0`. TR-01 defaults \(q_0=1\), \(\varepsilon=10^{-2}\), \(k=1\), \(a=1\). |
| Native dimensions | `POPS_NATIVE_DIM=1` listed for the planner. Optional `run_native` refuses live reread. |
| Required capabilities | None (`requires = []`). KokkosSerial listed for the planner; MPI off. No h5py. Live HDF5 is ROMEO-only. |
| Configurations | Single manufactured TR-01 state. No solver, CFL, integrator, flux, reconstruction, or AMR. |
| Diagnostics | Task 2 volume-weighted L1/L2/L∞ of restored vs original `centers` and `q`. Equality of `components` (order) and `owner`. |
| Thresholds | Round-trip \(L^\infty=0\). Component order preserved. Empty `orders` with reason `npz stand-in / live HDF5 on ROMEO`. |
| Proves | numpy `.npz` serialization of a TR-01 HDF5-shaped `{centers, q, components, owner}` dict is an identity on arrays, component order, and owner. Report renderer accepts an IF-10 summary. |
| Does not prove | Live HDF5 collective I/O, `pops.output.read_hdf5`, MPI rank remapping, AMR rematerialization, bit-identical native restart, spatial/temporal order. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
