# TR-01 — 3-d oblique periodic advection sine

Annexe A.1 and §35.1 `01_advection_sine_oblique_3d`. The Case is Cartesian
3-d only. A 1-d or 2-d run is not this case.

| Field | Content |
|---|---|
| Identifier | `TR-01` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + \mathbf a\cdot\nabla q = 0\) with \(\mathbf a=(1,1,1)\). |
| Oracle | \(q(\mathbf x,t)=q_0+\varepsilon\sin(2\pi\mathbf k\cdot(\mathbf x-\mathbf a t))\) with \(q_0=1\), \(\varepsilon=10^{-2}\), \(\mathbf k=(1,2,3)\). Cell averages via 4-point Gauss–Legendre. `exact.py` does not read PoPS output. |
| Domain and boundaries | Periodic unit cube \([0,1]^3\). `POPS_NATIVE_DIM=3` required. |
| Parameters | \(T=1\) (one period). Resolutions \(N=16,32,64,128\). |
| Native dimensions | `POPS_NATIVE_DIM=3` only. No 1-d/2-d fallback. |
| Required capabilities | Cartesian uniform periodic cube. KokkosSerial. MUSCL/VanLeer + ScalarUpwind, SSPRK2. Public `pops.diagnostics` on the ConsumerGraph. |
| Configurations | Uniform \(N^3\) cells. Adaptive CFL \(0.15\) so \(\lvert\mathbf a\rvert_1\Delta t/\Delta x=0.45\). Formal spatial order 2. No AMR. |
| Diagnostics | Volume-weighted L1/L2/L∞ vs cell-averaged exact. Observed spatial order on four resolutions. Native Integral/Norm/MinMax/ConservationCheck. Per-run `provenance.json`. |
| Thresholds | Observed order \(\ge 1.8\). |
| Proves | 3-d oblique periodic translation on a Dim-3 native artifact. |
| Does not prove | The catalog Case in `run.py` is the 3-d cube only. |
| Resources | Local Dim-3 series. |
| Provenance | `pops.verification.provenance.v1` written next to the native output. |

Plan §11 obligatory variants live in `complement.py` and are launched with
`POPS_NATIVE_DIM` matching the variant rank:

```
POPS_NATIVE_DIM=1 python verification/machines/run_tr01_complement.py --dim 1
```

That catalog covers 1-d ±a, 2-d four velocities, 3-d axes / diagonal /
oblique, `U-C` / `U-F` / `A-S0` / `A-S2` / `A-DP` / `A-DT`, block sizes,
1/2/4 periods, axis permutations, and L1/L2/L∞, phase, amplitude, mass,
spectrum, argmax, \(E_{cf}\)/\(E_{bulk}\). Multi-rank MPI remains IF-01
(`mpi_world` on an MPI leaf). Evidence: `build/verification/tr01-complement/`.
