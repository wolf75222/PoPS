"""Native Δt order campaign for TM-02 Lie and Strang (FV-compatible Programs)."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from verification.pops_verify.case_authoring import load_sibling_module


def main() -> int:
    run = load_sibling_module(
        ROOT / "verification/cases/time/noncommuting_strang/run.py"
    )
    # Heun substeps + WENO5 so spatial error sits below the temporal series.
    # CFL: a2=3, n=32 ⇒ h=1/32, dt < h/a2 ≈ 0.0104. Use 0.48 CFL at the coarse dt.
    dts = (0.005, 0.0025, 0.00125)
    # Longer window so Lie O(Δt) splitting can separate from Heun O(Δt²).
    t_end = 0.4
    n_cells = 32
    reconstruction_kind = "weno5"
    for method in ("lie", "strang"):
        print(f"=== TM-02 {method} {reconstruction_kind} n={n_cells} ===")
        campaign = run.run_order_campaign(
            dts,
            t_end=t_end,
            n_cells=n_cells,
            method=method,
            reconstruction_kind=reconstruction_kind,
        )
        print("dts", campaign["dts"])
        print("linf", campaign["linf"])
        print("orders", campaign["orders"])
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
