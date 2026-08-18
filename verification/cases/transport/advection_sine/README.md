# TR-01 — Periodic advection sine (1-d / 2-d / canonical 3-d)

Annexe A.1, §9.2, and §35.1 `01_advection_sine_oblique_3d`. One case
authority dispatches 1-d and 2-d restrictions plus the canonical 3-d cube.
A 1-d or 2-d run is never labeled canonical 3-d.

| Field | Content |
|---|---|
| Identifier | `TR-01` |
| `verification_kind` | `code-verification` |
| `evidence_status` | `required` |
| Equations | \(\partial_t q + \mathbf a\cdot\nabla q = 0\). |
| Canonical 3-d | \(q=1+10^{-2}\sin(2\pi(x+2y+3z))\), \(\mathbf a=(1,1,1)\), \(T=1\) on \([0,1]^3\). |
| 1-d / 2-d | Natural restriction of the same formula to the present coordinates. |
| Oracle | Translation \(q(\mathbf x-\mathbf a t,0)\). Cell averages via 4-point Gauss–Legendre. `exact.py` does not read PoPS output. |
| Domain | Periodic unit interval / square / cube. Exact-rank leaf required. No silent dim fallback. |
| Parameters | \(T=1\) (one period) for the default configs. Order series \(N=16,32,64,128\). |
| Native dimensions | `POPS_NATIVE_DIM` ∈ {1,2,3} matching the request. Shared 1-d IF/AM runtime stays in `tr01_runtime.py` and does not call 3-d. |
| Required capabilities | Cartesian uniform periodic. KokkosSerial / OpenMP / MPI as requested. MUSCL/VanLeer + ScalarUpwind, SSPRK2. |
| Configurations | Canonical 3-d; 1-d ±a; 2-d x/y/diagonal/(1,0.37); 3-d axes/diagonal/oblique; U-C, U-F, A-S0, A-S2, A-DP, A-DT; blocks 8/16/32/64; periods 1/2/4. AMR variants are executable or `not-supported`; they are not a silent uniform substitute. |
| Diagnostics | Volume-weighted L1/L2/L∞ vs cell-averaged exact; phase; amplitude; \(\int q\,dV\); observed spatial order only from ≥4 native resolutions. |
| Thresholds | Observed order \(\ge 1.8\) for an announced order-2 uniform series. A 16/32 smoke is never an order pass. |
| Proves | Exact-rank periodic translation on the requested dimension and config, when a native series exists. |
| Does not prove | Exact-vs-exact, injected \(h^2\), finite-only, or order from fewer than four native resolutions. |
| Resources | ROMEO x64cpu, ≤2 nodes, no GPU in this step. 3-d \(n\ge256\) is not launched here. |
| Provenance | `pops.verification.provenance.v1` with truthful RUN_FIELDS from the CampaignRequest. |
| Visuals | Phase 8 `visual_data/` from native results only; no committed fixtures. |

```
POPS_NATIVE_DIM=3 python scripts/run_verification.py --suite pr --dimensions 3 \
  --max-nodes 1 --pops-native-dim 3 --cases TR-01 --mpi-mode off \
  --execution-space KokkosSerial --execute --output build/verification/tr01
```
