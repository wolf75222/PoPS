# EU-06 — Exact uniform-flow preservation

Constant Euler free stream on a periodic unit square. The exact solution is
the initial condition at every \(t\). A manufactured 1-cell leftover at the
mid-domain block-face / CF join has \(L^\infty\) equal to the bump amplitude.
Native compile is optional.

| Field | Content |
|---|---|
| Identifier | `EU-06` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | 2-d gamma-law Euler, primitives \(W=(\rho,u,v,p)\). Conserved \((\rho,\rho u,\rho v,E)\). No sources. \(\gamma=1.4\). |
| Oracle | Spatially constant free stream \((\rho,u,v,p)=(1,1,0,1)\). Exact at every \(t\) is the IC. `exact.py` does not read PoPS output. |
| Domain and boundaries | 2-d periodic unit square \([0,1]^2\). Mid-domain face \(x=0.5\) is both a same-level block join and a static CF interface location. |
| Parameters | \(\rho=1\), \(u=1\), \(v=0\), \(p=1\), \(\gamma=1.4\). Manufactured leftover amplitude \(1/4\) (dyadic, so L∞ equals the amplitude exactly). Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=2`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 2-d, uniform, periodic, KokkosSerial, MPI off. HLLC/Rusanov + MUSCL when a native series exists. |
| Configurations | Single resolution \(n=32^2\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. |
| Diagnostics | Spatial constancy of the exact field. Task 2 volume-weighted L1/L2/L∞ of exact vs exact. L∞ leftover of a 1-cell block-face / CF bump versus the uniform state. |
| Thresholds | Exact field is invariant in \(t\). Exact-vs-exact L∞ = 0. 1-cell leftover L∞ equals the bump amplitude. Empty `orders` with reason containing `machine-zero free-stream`. |
| Proves | Uniform-flow oracle (spatially constant; time-invariant); leftover diagnostic detects a 1-cell interface bump at its amplitude; report renderer. |
| Does not prove | Native free-stream preservation, geometric conservation, AMR reflux, Poisson, coupling, MPI, HLLC vs Rusanov parity, observed spatial/temporal order. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
